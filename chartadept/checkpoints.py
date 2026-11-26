"""Check that an exported Hugging Face checkpoint has its weight files."""
import json
from pathlib import Path


def require_checkpoint(folder):
    folder = Path(folder).resolve()
    if not (folder / 'config.json').is_file():
        raise ValueError(f'Model checkpoint is missing config.json: {folder}')
    for name in ('model.safetensors.index.json', 'pytorch_model.bin.index.json'):
        index = folder / name
        if index.is_file():
            shards = set(json.loads(index.read_text())['weight_map'].values())
            missing = sorted(name for name in shards if not (folder / name).is_file())
            if not shards or missing:
                raise ValueError(f'Incomplete model checkpoint at {folder}: {missing}')
            return folder
    if any((folder / name).is_file() for name in ('model.safetensors', 'pytorch_model.bin')):
        return folder
    raise ValueError(f'No exported model weights at {folder}')
