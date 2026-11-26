"""Schedule one pass over the questions and exclude failed rollout groups."""
from collections import defaultdict
import json
from math import gcd
from pathlib import Path


class SkipTrajectoryError(RuntimeError):
    """A rollout failure that excludes the question group from the update."""


def checkpoint_steps(maximum):
    return [s for s in (4, 8, 12, 16) if s <= maximum] + list(range(26, maximum + 1, 10))


def sample_failure(error):
    # GPU/worker/collective failures require recovery of the training process.
    text = str(error).lower()
    if any(s in text for s in ('out of memory', 'cuda', 'nccl', 'actor died', 'worker died')):
        return False
    if isinstance(error, (FileNotFoundError, ValueError, KeyError, IndexError, TypeError)):
        return True
    return isinstance(error, RuntimeError) and any(s in text for s in (
        'maximum context length', 'max_model_len', 'prompt length', 'token id',
        'image size', 'image decode', 'cannot identify image'))


def valid_group_indices(uids, results, k=8):
    groups = defaultdict(list)
    for i, uid in enumerate(uids):
        groups[str(uid)].append(i)
    keep, skipped = [], []
    for uid, indices in groups.items():
        assert len(indices) == k, (uid, len(indices))
        failed = [i for i in indices if results[i].get('skip_trajectory')]
        if failed:
            skipped.append({'uid': uid, 'indices': indices, 'failed_indices': failed,
                'errors': [results[i] for i in failed]})
        else:
            keep.extend(indices)
    return keep, skipped


def build_partial(collector, world_size=96, optimizer_steps=1):
    """Keep complete groups divisible by DP size and native minibatch count."""
    from verl import DataProto
    import numpy as np
    divisor = world_size * optimizer_steps
    quantum = divisor // gcd(divisor, collector.k)
    count = min(collector.target_groups, collector.eligible_groups) // quantum * quantum
    if not count:
        return None, None, 0
    size = count * collector.k
    batch = DataProto.concat(collector.batches).select_idxs(np.arange(size, dtype=np.int64))
    extras = {key: np.concatenate([c[key] for c in collector.columns])[:size]
        for key in collector.columns[0]}
    assert len(batch) % divisor == 0
    return batch, extras, count


def record_skips(path, batch, skipped, step):
    if not skipped:
        return
    from chartadept.io import now
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        for group in skipped:
            first = group['indices'][0]
            stream.write(json.dumps({'step': step,
                'question_id': batch.non_tensor_batch['extra_info'][first]['question_id'],
                'skipped_trajectories': len(group['indices']),
                'failed_trajectories': len(group['failed_indices']),
                'errors': group['errors'], 'reward_used': False,
                'complete_group_excluded': True, 'group_size': len(group['indices']),
                'complete_k8_excluded': len(group['indices']) == 8,
                'recorded_at': now()}, ensure_ascii=False) + '\n')


def join_valid_rewards(batch, specification, step):
    import numpy as np
    import ray
    from chartadept.rl.deferred_reward import attach_rewards
    refs = list(batch.non_tensor_batch['chart_reward_future'])
    try:
        results = ray.get(refs)
    except ray.exceptions.RayTaskError:
        # Resolve individual tasks to identify which trajectories failed.
        results = []
        for i, ref in enumerate(refs):
            try:
                results.append(ray.get(ref))
            except ray.exceptions.RayTaskError as error:
                if not isinstance(error.cause, SkipTrajectoryError):
                    raise
                results.append({'skip_trajectory': True, 'error': str(error.cause),
                    'stage': 'reward',
                    'trace_file': str(batch.non_tensor_batch['chart_trace_file'][i])})
    keep, skipped = valid_group_indices(batch.non_tensor_batch['uid'], results,
        specification['rollouts_per_prompt'])
    record_skips(specification['quarantine_file'], batch, skipped, step)
    good = batch.select_idxs(np.asarray(keep, dtype=np.int64))
    if keep:
        attach_rewards(good, [results[i] for i in keep])
    return good, skipped
