"""Restore full FSDP2 state with PyTorch's supported load-time resharding API."""
import json
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions, set_model_state_dict, set_optimizer_state_dict,
)


def install_full_state(model, optimizer, state):
    # The full source is CPU-mapped, but destination shards stay on their model
    # device. CPU offload here would make optimizer broadcast use CPU with NCCL.
    options = StateDictOptions(full_state_dict=True, cpu_offload=False, broadcast_from_rank0=True, strict=False)
    # PyTorch2.8's rank0 broadcast canonicalizes tied parameters and omits their
    # duplicate names from load_state_dict. Only those aliases may be missing;
    # they refer to the exact same restored Parameter object.
    canonical = {name for name, _ in model.named_parameters()}
    aliases = {name for name, _ in model.named_parameters(remove_duplicate=False)} - canonical
    # On rank0 an unbroadcast alias otherwise remains a CPU Tensor and is
    # incorrectly copied into its DTensor. Restore each shared object once.
    source_model = {name: value for name, value in state.get('model', {}).items() if name not in aliases}
    restored = set_model_state_dict(model, source_model, options=options)
    assert not restored.unexpected_keys and set(restored.missing_keys) <= aliases, restored
    set_optimizer_state_dict(model, optimizer, state.get('optimizer', {}), options=options)


def restore_worker(worker, path):
    path = Path(path)
    manifest = json.loads((path / 'topology-restore.json').read_text())
    assert manifest['target_world_size'] == dist.get_world_size()
    assert manifest['source_world_size'] == 64 and manifest['preserve_optimizer']
    state = torch.load(path / 'full-state.pt', map_location='cpu', weights_only=False, mmap=True) if dist.get_rank() == 0 else {}
    install_full_state(worker.actor_module_fsdp, worker.actor_optimizer, state)
    del state
    original = Path(manifest['source_checkpoint']) / 'actor'
    extra = torch.load(original / f'extra_state_world_size_64_rank_{dist.get_rank()}.pt',
                       map_location='cpu', weights_only=False)
    worker.checkpoint_manager.load_rng_state(extra['rng'])
    worker.actor_lr_scheduler.load_state_dict(extra['lr_scheduler'])
    dist.barrier()
    return manifest
