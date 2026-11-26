"""Keep native reward tasks running while old-policy probabilities are computed."""
import asyncio
import json
import os
from pathlib import Path
import random

import numpy as np
import ray
import torch
from tensordict import TensorDict
from verl import DataProto
from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AgentLoopWorker, auto_await


def attach_rewards(batch, results):
    """Attach actual results in the current row order, after any batch balancing."""
    assert len(results) == len(batch)
    keys = list(results[0]['reward_extra_info'])
    assert all(set(row['reward_extra_info']) == set(keys) for row in results)
    scores = torch.tensor([row['reward_score'] for row in results], dtype=torch.float32)
    assert torch.isfinite(scores).all()
    prompt_length = batch.batch['prompts'].shape[1]
    ends = batch.batch['attention_mask'][:, prompt_length:].sum(dim=1).long() - 1
    rewards = torch.zeros_like(batch.batch['response_mask'], dtype=torch.float32)
    rewards[torch.arange(len(batch)), ends] = scores
    batch.batch['rm_scores'] = rewards
    for key in keys:
        batch.non_tensor_batch[key] = np.array([row['reward_extra_info'][key] for row in results])
    records = np.empty(len(results), dtype=object)
    records[:] = [row['reward_extra_info'] for row in results]
    batch.non_tensor_batch['reward_extra_info'] = records
    batch.meta_info['reward_extra_keys'] = keys
    batch.non_tensor_batch.pop('chart_reward_future', None)
    return batch


def join_rewards(batch):
    return attach_rewards(batch, ray.get(list(batch.non_tensor_batch['chart_reward_future'])))


class DeferredAgentLoopWorker(AgentLoopWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        specification = os.environ.get('CHART_RL_PAIRED_CONFIG')
        self._chart_compact_pixels = bool(specification and json.loads(Path(specification).read_text()).get(
            'pixel_transport') == 'lossless-zlib-v1')
        self._chart_skip_sample_failures=bool(specification and json.loads(Path(specification).read_text()).get('skip_sample_failures'))

    async def _agent_loop_postprocess(self, output, **kwargs):
        try:
            return await super()._agent_loop_postprocess(output, **kwargs)
        except Exception as error:
            from chartadept.reward.epoch_policy import sample_failure
            if not (getattr(self,'_chart_skip_sample_failures',False) and kwargs.get('chart_training_sample')) or output.extra_fields.get('chart_runtime_invalid') or not sample_failure(error):
                raise
            from chartadept.rl.agent_loop import ChartAgentLoop
            fallback=ChartAgentLoop._invalid_output(self,{**kwargs,'chart_error_after_generation':True},
                'native_postprocess '+type(error).__name__+': '+str(error))
            return await super()._agent_loop_postprocess(fallback,**kwargs)

    def _postprocess(self, inputs, input_non_tensor_batch=None):
        # Native per-trajectory outputs retain their typed Tensor contract.
        # Pack only the final opaque DataProto rows before Ray returns them.
        result = super()._postprocess(inputs, input_non_tensor_batch)
        if self._chart_compact_pixels:
            from chartadept.rl.pixel_transport import pack_row
            rows = result.non_tensor_batch['multi_modal_inputs']
            compact = np.empty(rows.shape, dtype=object)
            compact[:] = [pack_row(row or {}) for row in rows]
            result.non_tensor_batch['multi_modal_inputs'] = compact
        return result

    async def _compute_score(self, output, prompts, responses, attention_mask, input_ids, position_ids, kwargs):
        if output.extra_fields.get('chart_runtime_invalid'):
            output.extra_fields['chart_reward_future'] = ray.put(output.extra_fields['chart_runtime_invalid'])
            return
        # Keep the reward future so group selection can resolve it after generation.
        assert output.reward_score is None and self.reward_loop_worker_handles
        data = DataProto(batch=TensorDict({'prompts': prompts, 'responses': responses,
            'attention_mask': attention_mask, 'input_ids': input_ids, 'position_ids': position_ids}, batch_size=1),
            non_tensor_batch={**{key: np.array([value]) for key, value in kwargs.items()},
                '__num_turns__': np.array([output.num_turns]),
                'tool_extra_fields': np.array([output.extra_fields], dtype=object)})
        handle = random.choice(self.reward_loop_worker_handles)
        output.extra_fields['chart_reward_future'] = handle.compute_score.remote(data)


class DeferredAgentLoopManager(AgentLoopManager):
    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = ray.remote(DeferredAgentLoopWorker)
        super().__init__(*args, **kwargs)

    @auto_await
    async def generate_sequences(self, prompts):
        output = await super().generate_sequences(prompts)
        specification = os.environ.get('CHART_RL_PAIRED_CONFIG')
        if specification and json.loads(Path(specification).read_text()).get('pixel_transport') == 'lossless-zlib-v1':
            from chartadept.rl.pixel_transport import payload_statistics
            from chartadept.io import write_json, now
            root = Path(os.environ['CHART_RL_METRICS_FILE']).parent
            record = {**payload_statistics(output), 'step': prompts.meta_info.get('global_steps'),
                      'validation': bool(prompts.meta_info.get('validate')), 'updated_at': now()}
            write_json(root / 'pixel-transport.json', record)
            with (root / 'pixel-transport-journal.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
        # Attach the resolved rewards before returning a validation batch.
        if prompts.meta_info.get('validate', False):
            results = await asyncio.gather(*list(output.non_tensor_batch['chart_reward_future']))
            attach_rewards(output, results)
        return output
