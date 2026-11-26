"""Validate saved native state before restarting the existing RL workload."""
import json
from pathlib import Path


def validate_checkpoint(checkpoint, world_size, prompt_batch=None):
    checkpoint = Path(checkpoint)
    assert (checkpoint / 'data.pt').stat().st_size > 0
    step = int(checkpoint.name.removeprefix('global_step_'))
    actor = checkpoint / 'actor'
    manifest_file = actor / 'topology-restore.json'
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text())
        assert manifest['target_world_size'] == world_size and manifest['preserve_optimizer']
        assert manifest['source_step'] == step and (actor / 'full-state.pt').stat().st_size > 0
        return manifest['source_prompt_batch']
    for kind in ('model', 'optim', 'extra_state'):
        for rank in range(world_size):
            assert (actor / f'{kind}_world_size_{world_size}_rank_{rank}.pt').stat().st_size > 0
    index = json.loads((actor / 'huggingface/model.safetensors.index.json').read_text())
    for filename in set(index['weight_map'].values()):
        assert (actor / 'huggingface' / filename).stat().st_size > 0
    return prompt_batch if prompt_batch is not None else (126 if world_size == 56 else 128)


def latest_checkpoint(directory, world_size):
    directory = Path(directory)
    step = int((directory / 'latest_checkpointed_iteration.txt').read_text().strip())
    checkpoint = directory / f'global_step_{step}'
    validate_checkpoint(checkpoint, world_size)
    return checkpoint
