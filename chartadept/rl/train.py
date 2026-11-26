"""Run the VERL trainer across the selected Ray GPU nodes."""
import json
import os
from pathlib import Path

import hydra
import ray
from omegaconf import OmegaConf
from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import TaskRunner, run_ppo
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role

from chartadept.io import now
from chartadept.ray_runtime import reserve_verl_nodes


class ClusterPoolManager(ResourcePoolManager):
    def create_resource_pool(self):
        super().create_resource_pool()
        for pool in self.resource_pool_dict.values():
            reserve_verl_nodes(pool)


class ChartPPOTrainer(RayPPOTrainer):
    def _create_dataloader(self, *args, **kwargs):
        super()._create_dataloader(*args, **kwargs)
        paired = os.environ.get('CHART_RL_PAIRED_CONFIG')
        if paired and json.loads(Path(paired).read_text()).get('full_pool_pass'):
            from chartadept.rl.data import keep_tail
            self.train_dataloader = keep_tail(self.train_dataloader)
            print('Full-pool train batches including tail:', len(self.train_dataloader), flush=True)

    def fit(self):
        from chartadept.gspo.training import fit
        return fit(self)

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        # Training metadata arrives as NumPy columns; the official JSONL writer
        # indexes those columns into NumPy scalars that json.dumps cannot encode.
        import numpy as np
        columns = {key: value.tolist() if isinstance(value, np.ndarray) else value
                   for key, value in reward_extra_infos_dict.items()}
        return super()._dump_generations(inputs, outputs, gts, scores, columns, dump_path)


class ChartTaskRunner(TaskRunner):
    def init_resource_pool_mgr(self, config):
        manager = super().init_resource_pool_mgr(config)
        return ClusterPoolManager(manager.resource_pool_spec, manager.mapping)

    def add_actor_rollout_worker(self, config):
        from chartadept.rl.worker import ChartActorWorker
        _, group = super().add_actor_rollout_worker(config)
        self.role_worker_mapping[Role.ActorRollout] = ray.remote(ChartActorWorker)
        return ChartActorWorker, group

    def run(self, config):
        from numbers import Number
        from chartadept.rl.transport import install
        import verl.trainer.main_ppo as official_runner
        if os.environ.get('CHART_RL_PAIRED_CONFIG'):
            # The frozen SFT reference supplies drift diagnostics, not KL loss.
            import verl.trainer.ppo.ray_trainer as native_trainer
            official_runner.need_reference_policy = lambda _: True
            native_trainer.need_reference_policy = lambda _: True
            from chartadept.rl.low_port_server import LowPortServer
            from verl.workers.rollout.vllm_rollout import vllm_async_server
            vllm_async_server.vLLMHttpServer = LowPortServer
        from verl.utils.tracking import Tracking
        install()
        official_runner.RayPPOTrainer = ChartPPOTrainer
        original = Tracking.log
        metric_file = Path(os.environ['CHART_RL_METRICS_FILE'])
        metric_file.parent.mkdir(parents=True, exist_ok=True)

        def log(tracker, data, step, backend=None):
            original(tracker, data, step, backend)
            values = {key: float(value) for key, value in data.items() if isinstance(value, Number)}
            with metric_file.open('a') as stream:
                stream.write(json.dumps({'step': step, 'metrics': values, 'timestamp': now(),
                                         'phase': os.environ['CHART_RL_PHASE']}) + '\n')

        Tracking.log = log
        return super().run(config)


def resolve_configuration(config):
    config = migrate_legacy_reward_impl(config)
    # Ray workers do not inherit the driver's environment interpolation context.
    OmegaConf.resolve(config)
    return config


@hydra.main(config_path=None, config_name=None, version_base=None)
def main(config):
    config = resolve_configuration(config)
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(ChartTaskRunner))


if __name__ == '__main__':
    main()
