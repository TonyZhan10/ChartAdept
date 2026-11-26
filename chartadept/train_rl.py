"""Run outcome-only GSPO from the final SFT checkpoint."""
import argparse
import asyncio
from ipaddress import IPv4Address
import json
from math import gcd
import os
from pathlib import Path
import socket
import sys

from chartadept.io import ROOT, RUN_ROOT, output_directory, read_json, write_json


def configuration(stage, output=None, model=None, data=None, resume=None, updates=None):
    if stage != 'gspo':
        raise ValueError('Only the GSPO post-training stage is supported')
    import yaml
    value = yaml.safe_load((ROOT / f'configs/{stage}.yaml').read_text())
    output = output_directory(output or RUN_ROOT / stage)
    data = Path(data or RUN_ROOT / 'rl-data').resolve()
    value['data'].update(train_files=str(data / 'train.parquet'), val_files=str(data / 'development.parquet'))
    model_path = Path(model) if model else ROOT / value['actor_rollout_ref']['model']['path']
    value['actor_rollout_ref']['model']['path'] = str(model_path.resolve())
    value['actor_rollout_ref']['rollout']['agent']['agent_loop_config_path'] = str(ROOT / 'configs/agent.yaml')
    value['reward']['custom_reward_function']['path'] = str(ROOT / 'chartadept/rl/reward.py')
    value['trainer'].update(default_local_dir=str(output / 'checkpoints'),
        rollout_data_dir=str(output / 'training-rollouts'), validation_data_dir=str(output / 'validation'))
    value['global_profiler']['save_path'] = str(output / 'profile')
    value['ray_kwargs']['ray_init']['runtime_env']['py_executable'] = sys.executable
    if resume:
        value['trainer'].update(resume_mode='resume_path', resume_from_path=str(Path(resume).resolve()))
    if updates is not None:
        value['trainer']['total_training_steps'] = updates
    return value


def example_settings(config, nodes, gpus_per_node, updates):
    world = nodes * gpus_per_node
    samples = 8
    # VERL accepts mini-batch size in prompts, then multiplies by K and shards it over ranks.
    prompts_per_minibatch = world // gcd(world, samples)
    target = 4 * prompts_per_minibatch
    config['trainer'].update(nnodes=nodes, n_gpus_per_node=gpus_per_node, total_training_steps=updates,
        val_before_train=False, test_freq=-1, logger=['console'])
    config['data'].update(train_batch_size=2 * target, val_batch_size=8, val_max_samples=8,
        max_prompt_length=2048, max_response_length=512, shuffle=False, dataloader_num_workers=0)
    model = config['actor_rollout_ref']
    model['actor']['ppo_mini_batch_size'] = prompts_per_minibatch
    model['actor']['optim']['lr_warmup_steps'] = 0
    model['rollout'].update(n=samples, nnodes=nodes, n_gpus_per_node=gpus_per_node, temperature=1.0, top_p=1.0,
        gpu_memory_utilization=.22, enforce_eager=True, max_model_len=4096,
        max_num_batched_tokens=4096, max_num_seqs=8)
    model['rollout']['agent']['num_workers'] = min(world, 8)
    config['reward']['num_workers'] = min(world, 8)


async def run(stage, config, output, api_config, broker_host, broker_port, startup_check=False,
              scaled_example=False, numeric_rewards=False):
    from aiohttp import ClientSession
    from chartadept.reward.broker import run as reward_service
    import yaml

    output.mkdir(parents=True, exist_ok=True)
    recipe = read_json(ROOT / f'configs/{stage}-recipe.json')
    maximum = config['trainer']['total_training_steps']
    resume_path = config['trainer'].get('resume_from_path')
    recipe.update(output=str(output), updates=maximum, initial_model=config['actor_rollout_ref']['model']['path'],
        full_pool_updates=maximum, development_steps=list(range(0, maximum + 1, 4)),
        checkpoint_steps=list(range(4, maximum + 1, 4)), checkpoint_directory=config['trainer']['default_local_dir'],
        epoch_state_root=str(output / 'epoch-control'),
        resume_coverage_path=str(Path(resume_path) / 'epoch-coverage.json') if resume_path else
            str(output / 'initial-coverage.json'),
        resume_step=int(Path(resume_path).name.removeprefix('global_step_')) if resume_path else 0,
        quarantine_file=str(output / 'skipped-trajectories.jsonl'))
    if scaled_example:
        counts = read_json(Path(config['data']['train_files']).parent / 'ready.json')
        candidate_batch = config['data']['train_batch_size']
        rollout = config['actor_rollout_ref']['rollout']
        world = config['trainer']['nnodes'] * config['trainer']['n_gpus_per_node']
        minimum_groups = 4 * world // gcd(4 * world, rollout['n'])
        required = max(candidate_batch, (maximum - recipe['resume_step'] - 1) * candidate_batch + minimum_groups)
        if counts['train'] < required:
            raise ValueError(f"The {world}-GPU topology needs at least {required} questions to reach {maximum} "
                             f"updates in one pass. The prepared pool has {counts['train']}. "
                             "Use a larger question pool or request fewer updates.")
        target = 4 * config['actor_rollout_ref']['actor']['ppo_mini_batch_size']
        recipe.update(scaled_example=True, train_qa=counts['train'], full_pool_qa=counts['train'],
            development_qa=counts['development'], warmup_steps=0, planned_total_updates=maximum,
            max_generated_tokens=config['data']['max_response_length'], temperature=rollout['temperature'],
            top_p=rollout['top_p'], training_gpus=world,
            global_prompt_batch=target, target_effective_groups=target, candidate_groups_per_batch=candidate_batch,
            max_candidate_batches_per_update=4, rollouts_per_prompt=rollout['n'],
            rollout_padding_workers=rollout['agent']['num_workers'])
    write_json(output / 'recipe.json', recipe)
    api = read_json(api_config) if api_config else {}
    if not numeric_rewards and not api.get('model'):
        raise ValueError('The semantic reward configuration needs a model name.')
    options = {'reward_mode': 'outcome_only', 'numeric_rewards': numeric_rewards,
        'model': api.get('model', 'numeric-reference'), 'credential_config': str(api_config.resolve()) if api_config else None,
        'base_url': api.get('base_url'), 'rpm': 500, 'concurrency': 256,
        'development_file': config['data']['val_files'], 'development_qa': recipe['development_qa'],
        'task_reward_enabled': True, 'trust_env': False}
    write_json(output / 'broker-config.json', options)
    env = dict(os.environ, CHARTADEPT_ROOT=str(ROOT), PYTHONPATH=str(ROOT),
        CHART_RL_PAIRED_CONFIG=str(output / 'recipe.json'), CHART_RL_TOOL_PYTHON=sys.executable,
        CHART_RL_ROLLOUT_ROOT=str(output / 'rollouts'),
        CHART_RL_WITNESS_ROOT=str(output / 'ranks'), CHART_RL_PHASE='agentic_gspo',
        CHART_RL_METRICS_FILE=str(output / 'metrics.jsonl'),
        CHART_RL_TENSORBOARD_DIR=str(output / 'tensorboard'), TOKENIZERS_PARALLELISM='false')
    args = argparse.Namespace(output=output / 'reward', config=output / 'broker-config.json',
        port=broker_port, host=broker_host, owner_pid=os.getpid(), owner_create_time=None)
    service = asyncio.create_task(reward_service(args))
    process = None
    try:
        record = args.output / 'broker-process.json'
        async with ClientSession(trust_env=False) as session:
            for _ in range(120):
                if service.done():
                    service.result()
                if record.is_file():
                    broker = read_json(record)
                    if broker['pid'] == os.getpid():
                        async with session.get(broker['url'] + '/health') as response:
                            response.raise_for_status()
                        break
                await asyncio.sleep(.5)
            else:
                raise RuntimeError('Training broker did not become ready')
        env['CHART_RL_JUDGE_URL'] = broker['url']
        if not startup_check:
            from chartadept.ray_runtime import verify_broker
            await verify_broker(broker['url'])

        # Resolve the driver's paths and reward URL before creating any VERL workers.
        worker_env = config['ray_kwargs']['ray_init']['runtime_env']['env_vars']
        for key, value in list(worker_env.items()):
            if isinstance(value, str) and value.startswith('${oc.env:'):
                worker_env[key] = env[value[len('${oc.env:'):-1]]
        for key in ('CHART_RL_TOOL_PYTHON', 'CHARTADEPT_PAGEABLE_BUFFERS', 'NCCL_SOCKET_IFNAME',
                    'GLOO_SOCKET_IFNAME', 'PYTHONDONTWRITEBYTECODE', 'HF_HOME', 'HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE'):
            if key in env:
                worker_env[key] = env[key]
        config['ray_kwargs']['ray_init']['address'] = env.get('CHART_RL_RAY_ADDRESS', 'auto')
        path = output / 'effective.yaml'
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        check_args = ('--cfg', 'job', '--resolve') if startup_check else ()
        process = await asyncio.create_subprocess_exec(sys.executable, '-m', 'chartadept.rl.train',
            '--config-path', str(output), '--config-name', 'effective',
            *check_args, f'hydra.run.dir={output / "hydra"}', env=env, cwd=ROOT)
        completion = asyncio.create_task(process.wait())
        done, _ = await asyncio.wait([completion, service], return_when=asyncio.FIRST_COMPLETED)
        if service in done:
            service.result()
            raise RuntimeError('The reward broker stopped during training.')
        if completion.result():
            raise RuntimeError(f'Trainer exited with status {completion.result()}')
        if startup_check:
            print('GSPO startup check passed. The reward broker is ready and the VERL configuration resolved.')
    finally:
        if process and process.returncode is None:
            process.terminate()
            await process.wait()
        service.cancel()
        await asyncio.gather(service, return_exceptions=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['gspo'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--api-config', type=Path)
    parser.add_argument('--broker-host', default=os.environ.get('CHARTADEPT_BROKER_HOST'),
                        help='Driver IPv4 address reachable from all Ray nodes')
    parser.add_argument('--broker-port', type=int, default=int(os.environ.get('CHARTADEPT_BROKER_PORT', '0')))
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--updates', type=int, help='Override the number of rollout updates')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--startup-check', action='store_true',
        help='Start the reward broker and resolve the native trainer configuration')
    parser.add_argument('--nodes', type=int, help='Number of distinct nodes for the scaled Ray example')
    parser.add_argument('--gpus-per-node', type=int, help='GPUs per node for the scaled Ray example')
    parser.add_argument('--numeric-rewards', action='store_true', help='Use scalar numeric references for terminal correctness')
    args = parser.parse_args()
    if (args.nodes is None) != (args.gpus_per_node is None):
        parser.error('Supply both --nodes and --gpus-per-node for the scaled Ray example.')
    if args.nodes is not None and (args.nodes < 2 or args.gpus_per_node < 1):
        parser.error('Ray training requires at least two nodes and one GPU per node.')
    if args.updates is not None and args.updates < 1:
        parser.error('--updates must be positive')
    output = (args.output or RUN_ROOT / args.stage).resolve()
    data = (args.data or RUN_ROOT / 'rl-data').resolve()
    try:
        config = configuration(args.stage, output, args.model, data, args.resume, args.updates)
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    if args.nodes:
        example_settings(config, args.nodes, args.gpus_per_node, args.updates or 3)
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return
    if not (args.numeric_rewards or args.api_config):
        parser.error('Supply --api-config for semantic rewards or --numeric-rewards for numeric targets')
    if args.startup_check and not args.output:
        parser.error('--startup-check requires a separate --output')
    from chartadept.checkpoints import require_checkpoint
    try:
        require_checkpoint(config['actor_rollout_ref']['model']['path'])
    except ValueError as error:
        parser.error(str(error))
    if not (data / 'ready.json').is_file():
        parser.error('Supply the prepared RL data directory with --data')
    if args.numeric_rewards and not read_json(data / 'ready.json').get('numeric_rewards'):
        parser.error('--numeric-rewards requires a prepared dataset with numeric training and development targets')
    api_config = args.api_config.resolve() if not args.numeric_rewards else None
    os.chdir(ROOT)
    import ray
    try:
        if not args.startup_check or not args.broker_host:
            ray.init(address=os.environ.get('CHART_RL_RAY_ADDRESS', 'auto'), log_to_driver=False)
        host = IPv4Address(socket.gethostbyname(args.broker_host or ray.util.get_node_ip_address()))
        if host.is_loopback or host.is_unspecified:
            parser.error('Use a driver IPv4 address reachable from every Ray node as --broker-host.')
        asyncio.run(run(args.stage, config, output, api_config, str(host), args.broker_port,
            startup_check=args.startup_check, scaled_example=bool(args.nodes), numeric_rewards=args.numeric_rewards))
    finally:
        ray.shutdown()


if __name__ == '__main__':
    main()
