"""Publish progress for the GSPO training loop."""
import json
import os
from pathlib import Path

from chartadept.io import write_json, now


def publish_phase(step, stage):
    root = Path(os.environ['CHART_RL_METRICS_FILE']).parent
    value = {'step': step, 'stage': stage, 'started_at': now(), 'reward_probability_overlap': True}
    write_json(root / 'phase-progress.json', value)
    with (root / 'phase-events.jsonl').open('a') as stream:
        stream.write(json.dumps(value) + '\n')
