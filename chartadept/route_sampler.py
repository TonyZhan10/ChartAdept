"""Sample SFT trajectories by route using fixed quotas for each epoch."""
from collections import Counter
import json
from pathlib import Path
import random

WEIGHTS={'direct':24,'zoom_once':6,'python_once':6,'zoom_multiple':1,'zoom_python':3}


def make_plan(order,per_epoch,epochs=3,seed=42):
    groups={name:[i for i,row in enumerate(order) if row['bucket']==name] for name in WEIGHTS}
    assert all(groups.values())
    quotas={name:per_epoch*weight//40 for name,weight in WEIGHTS.items()}
    remainder=sorted(WEIGHTS,key=lambda name:-(per_epoch*WEIGHTS[name]%40))
    for name in remainder[:per_epoch-sum(quotas.values())]:quotas[name]+=1
    queues={name:[] for name in groups};positions={name:0 for name in groups}
    randoms={name:random.Random(seed+100003*i) for i,name in enumerate(groups)}
    plans=[];stats=[];seen=set()
    for epoch in range(epochs):
        indices=[]
        for name in groups:
            for _ in range(quotas[name]):
                if positions[name]==len(queues[name]):
                    queues[name]=list(groups[name]);randoms[name].shuffle(queues[name]);positions[name]=0
                indices.append(queues[name][positions[name]]);positions[name]+=1
        random.Random(seed+1000000+epoch).shuffle(indices)
        seen.update(indices);plans.append(indices)
        stats.append({'epoch':epoch+1,'reads':len(indices),'unique_qa':len(set(indices)),
            'cumulative_unique_qa':len(seen),
            'routes':{name:{'reads':quotas[name],
                'unique_qa':len({i for i in indices if order[i]['bucket']==name})} for name in groups}})
    totals=Counter(order[i]['bucket'] for indices in plans for i in indices)
    return {'seed':seed,'cache_rows':len(order),'epochs':plans,'epoch_summaries':stats,
        'per_epoch':per_epoch,'total_reads':per_epoch*epochs,'covered_unique_qa':len(seen),
        'target_weights':{name:weight/40 for name,weight in WEIGHTS.items()},
        'route_totals':dict(totals),'route_quotas':quotas}


def install(plan_path,expected_rows):
    """Replace only the training sampler; Accelerate still shards global batches."""
    from torch.utils.data import Sampler
    from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
    plan=json.loads(Path(plan_path).read_text())
    assert plan['cache_rows']==expected_rows

    class RouteSampler(Sampler):
        def __init__(self):self.epoch=0
        def __len__(self):return plan['per_epoch']
        def set_epoch(self,epoch):self.epoch=epoch
        def __iter__(self):return iter(plan['epochs'][self.epoch])

    sampler=RouteSampler()
    def get_sampler(trainer,train_dataset=None):
        assert len(train_dataset if train_dataset is not None else trainer.train_dataset)==expected_rows
        assert not trainer.args.group_by_length
        return sampler
    CustomSeq2SeqTrainer._get_train_sampler=get_sampler
    return sampler,plan
