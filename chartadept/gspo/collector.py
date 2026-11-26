"""Accumulate complete informative GSPO groups before one native update."""
import numpy as np

from chartadept.gspo.selection import select_batch


class GroupCollector:
    def __init__(self, target_groups=144, max_batches=4, k=16, epsilon=1e-6):
        self.target_groups, self.max_batches, self.k, self.epsilon = target_groups,max_batches,k,epsilon
        self.batches, self.columns, self.decisions = [],[],[]
        self.candidate_groups = self.eligible_groups = self.generation_batches = 0

    @property
    def ready(self):
        return self.eligible_groups >= self.target_groups

    def add(self, batch, columns, group_order):
        assert self.generation_batches < self.max_batches and not self.ready
        selected, extras, decision = select_batch(batch,columns,self.k,self.epsilon,group_order)
        self.generation_batches += 1
        self.candidate_groups += len(decision['groups'])
        self.eligible_groups += len(selected)//self.k
        self.decisions.extend(decision['groups'])
        if len(selected):
            # Per-candidate image lengths/token counts differ. Reconstruct them
            # on the final selected batch, preserving the common temperature.
            selected.meta_info = {'temperature':batch.meta_info['temperature']}
            self.batches.append(selected)
            self.columns.append(extras)

    def summary(self):
        return {'candidate_batches':self.generation_batches,'max_candidate_batches':self.max_batches,
            'candidate_groups':self.candidate_groups,'candidate_trajectories':self.candidate_groups*self.k,
            'eligible_groups':self.eligible_groups,'target_effective_groups':self.target_groups,
            'filtered_flat_groups':sum(row['reason']=='flat_reward' for row in self.decisions),
            'ready':self.ready,'k':self.k}

    def build(self):
        from verl import DataProto
        if not self.ready:
            raise RuntimeError('GROUP_FILTER_BUDGET_EXHAUSTED: fewer than target effective groups')
        size=self.target_groups*self.k
        batch=DataProto.concat(self.batches).select_idxs(np.arange(size,dtype=np.int64))
        extras={key:np.concatenate([columns[key] for columns in self.columns])[:size]
                for key in self.columns[0]}
        assert len(batch)==size and len(set(batch.non_tensor_batch['uid']))==self.target_groups
        return batch,extras
