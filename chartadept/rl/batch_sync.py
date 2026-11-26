"""Synchronize dynamic FSDP batches across the data-parallel group."""
import json
import os
from pathlib import Path

import torch.distributed as dist
from verl.utils.seqlen_balancing import prepare_dynamic_batch as _prepare_dynamic_batch


def prepare_dynamic_batch(data, max_token_len, **kwargs):
    # With SP=TP=1, the full-shard data-parallel group spans all ranks.
    if kwargs.get('dp_group') is None and dist.is_initialized():
        kwargs['dp_group'] = dist.group.WORLD
    batches, indices = _prepare_dynamic_batch(data, max_token_len, **kwargs)
    witness = os.environ.get('CHART_RL_WITNESS_ROOT')
    if witness and dist.is_initialized():
        folder = Path(witness).parent / 'batch-sync'
        folder.mkdir(exist_ok=True)
        record = {'rank': dist.get_rank(), 'world_size': dist.get_world_size(),
                  'rows': len(data), 'micro_batches': len(batches),
                  'max_token_len': max_token_len, 'explicit_dp_group': True}
        with (folder / f'rank-{dist.get_rank():02d}.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
    return batches, indices
