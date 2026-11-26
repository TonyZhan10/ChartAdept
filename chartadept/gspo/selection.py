"""Select complete GSPO groups with differing binary terminal rewards."""
from collections import Counter


def select_groups(group_ids, rewards, k=16, epsilon=1e-6):
    assert len(group_ids) == len(rewards)
    groups = {}
    for index, group_id in enumerate(group_ids):
        groups.setdefault(str(group_id), []).append(index)
    keep, decisions = [], []
    for group_id, indices in groups.items():
        assert len(indices) == k, (group_id, len(indices), k)
        scores = [float(rewards[i]['score']) for i in indices]
        assert all(score in (0., 1.) for score in scores)
        span = max(scores) - min(scores)
        accepted = span > epsilon
        if accepted:
            keep.extend(indices)
        decisions.append({'group_id': group_id, 'indices': indices,
            'final_reward_span': span, 'kept': accepted,
            'reason': 'terminal_reward_difference' if accepted else 'flat_reward'})
    return {'keep_indices': keep, 'groups': decisions,
        'counts': dict(Counter(row['reason'] for row in decisions)),
        'k': k, 'epsilon': epsilon, 'reward_changed': False}


def select_batch(batch, reward_columns, k=16, epsilon=1e-6, group_order=None):
    import numpy as np
    rewards = [{'score': score} for score in reward_columns['score']]
    decision = select_groups(batch.non_tensor_batch['uid'], rewards, k, epsilon)
    if group_order is not None:
        by_uid = {row['group_id']: row for row in decision['groups']}
        assert set(map(str, group_order)) == set(by_uid)
        decision['keep_indices'] = [i for uid in group_order if by_uid[str(uid)]['kept']
                                    for i in by_uid[str(uid)]['indices']]
    indices = np.asarray(decision['keep_indices'], dtype=np.int64)
    selected = batch.select_idxs(indices)
    columns = {key: np.asarray(values)[indices] for key, values in reward_columns.items()}
    return selected, columns, decision
