"""Expose the replica sampling seed in VERL's rollout configuration."""
from dataclasses import dataclass

from verl.workers.config import RolloutConfig


@dataclass
class ChartRolloutConfig(RolloutConfig):
    seed: int = 0
