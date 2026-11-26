"""Retain the first recovery checkpoint and all four-step saves in this run."""


def protect_history(worker):
    import json
    import os
    from pathlib import Path
    config = os.environ.get('CHART_RL_PAIRED_CONFIG')
    retain_all = bool(config and json.loads(Path(config).read_text()).get('retain_all_checkpoints'))
    manager = worker.checkpoint_manager
    if getattr(manager, '_chart_history_protected', False):
        return {'retained_checkpoints': None if retain_all else 17, 'rank': worker._chart_record['rank']}
    original = manager.save_checkpoint

    def save(local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        return original(local_path=local_path, hdfs_path=hdfs_path, global_step=global_step,
            max_ckpt_to_keep=None if retain_all else 17)

    manager.save_checkpoint = save
    manager._chart_history_protected = True
    return {'retained_checkpoints': None if retain_all else 17, 'rank': worker._chart_record['rank']}


def protect_in_actor(actor):
    worker = next(w for w in actor.worker_dict.values() if hasattr(w, '_chart_steps'))
    return protect_history(worker)
