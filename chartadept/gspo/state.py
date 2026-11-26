"""Save and restore question coverage at GSPO checkpoints."""
import json
from pathlib import Path
import time

from chartadept import io as progress


class EpochCoverage:
    def __init__(self, spec, step):
        self.spec = spec
        self.total = spec['full_pool_qa']
        self.root = Path(spec['epoch_state_root'])
        self.root.mkdir(parents=True, exist_ok=True)
        checkpoint = Path(spec['checkpoint_directory']) / f'global_step_{step}'
        source = checkpoint / 'epoch-coverage.json'
        if not source.exists():
            source = Path(spec['resume_coverage_path'])
        saved = (json.loads(source.read_text()) if source.exists() else
                 {'checkpoint_step': 0, 'question_ids': []})
        assert saved['checkpoint_step'] == step
        self.seen = set(saved['question_ids'])
        self.skipped_groups = saved.get('skipped_groups', 0)
        self.failed_trajectories = saved.get('failed_trajectories', 0)
        self.passes_without_update = saved.get('passes_without_update', 0)
        self.publish(status='restored_from_checkpoint', checkpoint_step=step)

    @property
    def finished(self):
        return len(self.seen) == self.total

    def consume(self, ids):
        assert len(set(ids)) == len(ids) and not self.seen.intersection(ids), (
            'Resumed data iterator repeated a QA within the same epoch')
        self.seen.update(ids)
        assert len(self.seen) <= self.total

    def snapshot(self, step):
        return {'checkpoint_step': step, 'question_ids': sorted(self.seen),
            'visited_qa': len(self.seen), 'total_qa': self.total,
            'skipped_groups': self.skipped_groups,
            'failed_trajectories': self.failed_trajectories,
            'passes_without_update': self.passes_without_update,
            'full_epoch_complete': self.finished, 'updated_at': progress.now()}

    def publish(self, skipped_groups=0, failed_trajectories=0, optimizer_update=None, **changes):
        self.skipped_groups += skipped_groups
        self.failed_trajectories += failed_trajectories
        self.passes_without_update += optimizer_update is False
        progress.write_json(self.root / 'coverage-status.json', {
            'visited_qa': len(self.seen), 'total_qa': self.total,
            'full_epoch_complete': self.finished, 'skipped_groups': self.skipped_groups,
            'skipped_group_trajectories': self.skipped_groups * self.spec.get('rollouts_per_prompt',8),
            'failed_trajectories': self.failed_trajectories,
            'passes_without_update': self.passes_without_update,
            'updated_at': progress.now(), **changes})


def save_checkpoint(trainer, coverage, root):
    from chartadept.gspo.continuation import validate_checkpoint
    step = trainer.global_steps
    directory = Path(trainer.config.trainer.default_local_dir)
    path = directory / f'global_step_{step}'
    old_directory = trainer.config.trainer.default_local_dir
    existing = path / 'epoch-coverage.json'
    if existing.exists() or (path/'data.pt').exists():
        try:
            validate_checkpoint(path, trainer.config.trainer.nnodes * trainer.config.trainer.n_gpus_per_node)
            complete = True
        except (AssertionError, FileNotFoundError, ValueError):
            complete = False
        if complete:
            if existing.exists() and json.loads(existing.read_text())['question_ids'] == sorted(coverage.seen):
                if coverage.finished:
                    progress.write_json(coverage.root / 'final-checkpoint.json',
                        {'checkpoint': str(path), 'step': step, 'passed': True})
                return
            # An all-flat tail advances the data cursor without an optimizer step.
            assert coverage.finished
            directory = directory / 'epoch-final'
            path = directory / f'global_step_{step}'
            trainer.config.trainer.default_local_dir = str(directory)
        else:
            path.rename(path.with_name(path.name + '.incomplete-' + str(time.time_ns())))
    path.mkdir(parents=True, exist_ok=True)
    progress.write_json(path / 'epoch-coverage.json', coverage.snapshot(step))
    try:
        trainer._save_checkpoint()
    finally:
        trainer.config.trainer.default_local_dir = old_directory
    if coverage.finished:
        progress.write_json(coverage.root / 'final-checkpoint.json',
            {'checkpoint': str(path), 'step': step, 'passed': True})
