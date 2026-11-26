"""VERL FSDP2 actor with kernel checks and optimizer-step measurements."""
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

from chartadept.io import write_json, now

PROBES = {
    'vision': 'model.visual.blocks.0.attn.qkv.weight',
    'merger': 'model.visual.merger.linear_fc1.weight',
    'language': 'model.language_model.layers.0.self_attn.q_proj.weight',
}


def local(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def cuda_witness():
    from flash_attn import flash_attn_func
    import torch.nn.functional as functional
    device = torch.cuda.current_device()
    with torch.random.fork_rng(devices=[device]):
        torch.manual_seed(1729)
        left = torch.randn(64, 64, device='cuda', dtype=torch.bfloat16)
        right = torch.randn(64, 64, device='cuda', dtype=torch.bfloat16)
        product = left @ right
        reference = left.float() @ right.float()
        torch.testing.assert_close(product.float(), reference, atol=.25, rtol=.025)
        q, k, v = [torch.randn(1, 128, 4, 64, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
        actual = flash_attn_func(q, k, v, causal=True)
        expected = functional.scaled_dot_product_attention(q.transpose(1, 2).float(), k.transpose(1, 2).float(),
                                                           v.transpose(1, 2).float(), is_causal=True).transpose(1, 2)
        torch.testing.assert_close(actual.float(), expected, atol=.03, rtol=.04)
        marker = torch.tensor(float(dist.get_rank() + 1), device='cuda')
        dist.all_reduce(marker)
        world = dist.get_world_size()
        assert marker.item() == world * (world + 1) // 2
        return {'bf16_matmul': True, 'flash_attention': True, 'world_size': dist.get_world_size(),
                'nccl_rank_sum': marker.item(), 'matmul_max_abs_error': (product.float() - reference).abs().max().item(),
                'attention_max_abs_error': (actual.float() - expected).abs().max().item()}


class ChartActorWorker(AsyncActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        if os.environ.get('CHART_RL_PAIRED_CONFIG'):
            import random
            import numpy as np
            seed = json.loads(Path(os.environ['CHART_RL_PAIRED_CONFIG']).read_text())['seed'] + self.rank
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        from verl.workers.actor import dp_actor
        from verl.workers import fsdp_workers
        from chartadept.rl.batch_sync import prepare_dynamic_batch
        dp_actor.prepare_dynamic_batch = prepare_dynamic_batch
        native_apply = fsdp_workers.apply_fsdp2
        if os.environ.get('CHARTADEPT_PAGEABLE_BUFFERS') == '1':
            def apply_pageable(model, kwargs, config):
                policy = kwargs['offload_policy']
                if policy is not None:
                    policy.pin_memory = False
                return native_apply(model, kwargs, config)

            fsdp_workers.apply_fsdp2 = apply_pageable
        try:
            result = super().init_model()
        finally:
            fsdp_workers.apply_fsdp2 = native_apply
        compact_pixels = False
        if os.environ.get('CHART_RL_PAIRED_CONFIG'):
            compact_pixels = json.loads(Path(os.environ['CHART_RL_PAIRED_CONFIG']).read_text()).get(
                'pixel_transport') == 'lossless-zlib-v1'
        if compact_pixels:
            from chartadept.rl.pixel_transport import install as install_pixels
            install_pixels()
        if not self._is_actor:
            if self._is_ref:
                model = Path(self.config.model.path)
                index = json.loads((model / 'model.safetensors.index.json').read_text())['weight_map']
                parameters = dict(self.ref_module_fsdp.named_parameters())
                checks = {}
                for component, name in PROBES.items():
                    parameter = parameters[name]
                    if isinstance(parameter, DTensor):
                        shape, offset = compute_local_shape_and_global_offset(parameter.shape, parameter.device_mesh, parameter.placements)
                        slices = tuple(slice(start, start + size) for start, size in zip(offset, shape))
                    else:
                        slices = tuple(slice(0, size) for size in parameter.shape)
                    actual = local(parameter).detach().reshape(-1)[:64].float().cpu()
                    with safe_open(model / index[name], framework='pt', device='cpu') as source:
                        expected = source.get_slice(name)[slices].reshape(-1)[:64].float()
                    assert torch.equal(actual, expected), (dist.get_rank(), component, 'SFT reference mismatch')
                    checks[component] = {'exact_source_match': True, 'parameter': name}
                assert self.ref_policy.actor_optimizer is None
                write_json(Path(os.environ['CHART_RL_WITNESS_ROOT']) / f'ref-rank-{dist.get_rank():02d}.json',
                    {'rank': dist.get_rank(), 'world_size': dist.get_world_size(), 'checkpoint': str(model),
                     'source_checks': checks, 'reference_has_optimizer': False,
                     **({'pixel_transport_mode': 'lossless-zlib-v1'} if compact_pixels else {}),
                     'dynamic_batch_dp_group_sync': True, 'updated_at': now()})
            return result
        self._chart_steps = 0
        self._chart_start_step = 0
        self._chart_parameters = dict(self.actor_module_fsdp.named_parameters())
        self._chart_initial = {}
        self._chart_source_checks = {}
        model = Path(self.config.model.path)
        index = json.loads((model / 'model.safetensors.index.json').read_text())['weight_map']
        for component, name in PROBES.items():
            parameter = self._chart_parameters[name]
            tensor = local(parameter).detach()
            values = tensor.reshape(-1)[:64].float().cpu().clone()
            if isinstance(parameter, DTensor):
                shape, offset = compute_local_shape_and_global_offset(parameter.shape, parameter.device_mesh, parameter.placements)
                slices = tuple(slice(start, start + size) for start, size in zip(offset, shape))
            else:
                slices = tuple(slice(0, size) for size in parameter.shape)
            with safe_open(model / index[name], framework='pt', device='cpu') as source:
                expected = source.get_slice(name)[slices].reshape(-1)[:64].float()
            assert torch.equal(values, expected), (dist.get_rank(), component, 'SFT initialization mismatch')
            self._chart_initial[component] = values
            self._chart_source_checks[component] = {'parameter': name, 'values': values.tolist(),
                                                     'exact_source_match': True, 'local_shape': list(tensor.shape)}
        import ray
        self._chart_record = {'rank': dist.get_rank(), 'world_size': dist.get_world_size(), 'ip': ray.util.get_node_ip_address(),
                              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                              'gpu_name': torch.cuda.get_device_name(), 'checkpoint': str(model),
                              'global_parameters': sum(p.numel() for p in self._chart_parameters.values()),
                              'global_trainable_parameters': sum(p.numel() for p in self._chart_parameters.values() if p.requires_grad),
                              'source_checks': self._chart_source_checks, 'kernels': cuda_witness(),
                              'optimizer_steps': 0, 'phase': os.environ['CHART_RL_PHASE'],
                              'entropy_from_logits_with_chunking': bool(self.actor.config.entropy_from_logits_with_chunking),
                              'dynamic_batch_dp_group_sync': True,
                              'use_kl_loss': bool(self.actor.config.use_kl_loss),
                              'kl_loss_coefficient': float(self.actor.config.kl_loss_coef),
                              'updated_at': now()}
        if compact_pixels:
            from chartadept.rl.pixel_transport import status as pixel_status
            self._chart_record['pixel_transport'] = pixel_status()
        self._chart_write()
        from chartadept.rl.checkpoint_policy import protect_history
        protect_history(self)
        self.actor_optimizer.register_step_post_hook(self._chart_after_step)
        if os.environ.get('CHART_RL_PAIRED_CONFIG'):
            specification = json.loads(Path(os.environ['CHART_RL_PAIRED_CONFIG']).read_text())
            if specification.get('gspo_enabled'):
                from chartadept.gspo.actor import install, install_diagnostics
                install(self.actor, specification)
                install_diagnostics(self, specification)
                self._chart_record.update(gspo_native_sequence_loss=True,
                    optimizer_steps_per_rollout_update=4, global_token_mean_normalization=False,
                    sequence_mean_token_mean=True, reference_is_diagnostic_only=True)
                self._chart_write()
                return result
        if os.environ['CHART_RL_PHASE'] == 'gpu_preflight':
            original_update = self.actor.update_policy

            def checked_update(data):
                checked = 0
                for i, trace_file in enumerate(data.non_tensor_batch['chart_trace_file']):
                    witness = json.loads(Path(trace_file).with_name('token-witness.json').read_text())
                    response, mask = witness['response_ids'], witness['response_mask']
                    assert data.batch['responses'][i, :len(response)].tolist() == response
                    assert data.batch['response_mask'][i, :len(mask)].tolist() == mask
                    assert not data.batch['response_mask'][i, len(mask):].any()
                    prompt = witness['prompt_ids']
                    assert data.batch['prompts'][i, -len(prompt):].tolist() == prompt
                    multimodal = data.non_tensor_batch['multi_modal_inputs'][i]
                    assert multimodal['image_grid_thw'].shape[0] == witness['image_count']
                    checked += 1
                assert data.batch['position_ids'].shape[1] == 4
                self._chart_record.setdefault('actual_actor_token_checks', []).append(
                    {'rows': checked, 'sampled_ids_equal': True, 'observation_mask_equal': True,
                     'image_counts_equal': True, 'multimodal_position_channels': 4})
                self._chart_write()
                return original_update(data)

            self.actor.update_policy = checked_update
        return result

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        topology_restore = local_path and (Path(local_path) / 'topology-restore.json').exists()
        if topology_restore:
            from chartadept.rl.topology_restore import restore_worker
            restore_worker(self, local_path)
            result = None
        else:
            result = super().load_checkpoint(local_path, hdfs_path, del_local_after_load)
        if local_path is None or not self._is_actor:
            return result
        step = int(Path(local_path).parent.name.removeprefix('global_step_'))
        hf = Path(local_path) / 'huggingface'
        index = json.loads((hf / 'model.safetensors.index.json').read_text())['weight_map']
        checks = {}
        optimizer_checks = {}
        reference = torch.load(Path(local_path) / 'full-state.pt', map_location='cpu',
                               weights_only=False, mmap=True) if topology_restore else None
        for component, name in PROBES.items():
            parameter = self._chart_parameters[name]
            tensor = local(parameter).detach()
            if isinstance(parameter, DTensor):
                shape, offset = compute_local_shape_and_global_offset(parameter.shape, parameter.device_mesh, parameter.placements)
                slices = tuple(slice(start, start + size) for start, size in zip(offset, shape))
            else:
                slices = tuple(slice(0, size) for size in parameter.shape)
            values = tensor.reshape(-1)[:64].float().cpu().clone()
            with safe_open(hf / index[name], framework='pt', device='cpu') as source:
                expected = source.get_slice(name)[slices].reshape(-1)[:64].float()
            assert torch.equal(values, expected), (self.rank, component, 'Native resume does not match saved HF weights')
            if reference is not None:
                optimizer_checks[component] = {}
                for key in ('exp_avg', 'exp_avg_sq'):
                    actual = local(self.actor_optimizer.state[parameter][key]).detach().reshape(-1)[:64].cpu()
                    expected_moment = reference['optimizer']['state'][name][key][slices].reshape(-1)[:64]
                    assert torch.equal(actual, expected_moment), (self.rank, component, key, 'Adam resume mismatch')
                    optimizer_checks[component][key] = {'exact_source_match': True, 'values_checked': actual.numel()}
            self._chart_initial[component] = values
            checks[component] = {'exact_saved_hf_match': True, 'values': values.tolist()}
        optimizer_steps = [int(value['step'].item()) for value in self.actor_optimizer.state.values() if 'step' in value]
        factor = self._chart_record.get('optimizer_steps_per_rollout_update', 1)
        expected_optimizer_steps = step * factor
        assert optimizer_steps and min(optimizer_steps) == max(optimizer_steps) == expected_optimizer_steps
        self._chart_steps = self._chart_start_step = expected_optimizer_steps
        self._chart_record.update(optimizer_steps=expected_optimizer_steps, resume_checkpoint=str(local_path), resume_step=step,
            resume_checks=checks, loaded_optimizer_steps=sorted(set(optimizer_steps)),
            resume_optimizer_checks=optimizer_checks,
            loaded_scheduler_step=self.actor_lr_scheduler.last_epoch, updated_at=now())
        self._chart_write()
        del reference
        return result

    def _chart_write(self):
        write_json(Path(os.environ['CHART_RL_WITNESS_ROOT']) / f'rank-{dist.get_rank():02d}.json', self._chart_record)

    def _chart_after_step(self, optimizer, args, kwargs):
        self._chart_steps += 1
        interval = self._chart_record.get('optimizer_steps_per_rollout_update', 16)
        if self._chart_steps > self._chart_start_step + 2 and self._chart_steps % interval:
            return
        probes = {}
        for component, name in PROBES.items():
            parameter = self._chart_parameters[name]
            values = local(parameter).detach().reshape(-1)[:64].float().cpu()
            difference = (values - self._chart_initial[component]).abs()
            gradient = local(parameter.grad) if parameter.grad is not None else None
            probes[component] = {'parameter': name, 'values': values.tolist(),
                                 'sampled_values': values.numel(), 'empty_local_shard': values.numel() == 0,
                                 'changed_values': (difference > 0).sum().item(),
                                 'max_abs_change': difference.max().item() if difference.numel() else None,
                                 'finite': torch.isfinite(values).all().item(),
                                 'gradient_present': gradient is not None,
                                 'sample_gradient_finite': bool(gradient is not None and torch.isfinite(gradient.reshape(-1)[:64]).all())}
        self._chart_record.update(optimizer_steps=self._chart_steps, component_updates=probes,
                                  learning_rates=[group['lr'] for group in optimizer.param_groups], updated_at=now())
        if 'pixel_transport' in self._chart_record:
            from chartadept.rl.pixel_transport import status as pixel_status
            self._chart_record['pixel_transport'] = pixel_status()
        self._chart_write()
