"""Train ChartAdept with SFT and GSPO, then generate predictions."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from chartadept.checkpoints import require_checkpoint
from chartadept.io import ROOT, output_directory, read_json, read_jsonl_by_id, write_json

STAGES = ('sft', 'gspo', 'infer')


def model_checkpoint(work):
    if os.environ.get('BASE_MODEL'):
        return require_checkpoint(os.environ['BASE_MODEL'])
    repo = 'Qwen/Qwen3-VL-4B-Instruct'
    model = work / 'models' / repo.split('/')[1]
    try:
        return require_checkpoint(model)
    except ValueError:
        from huggingface_hub import snapshot_download
        snapshot_download(repo, local_dir=model, token=False,
                          allow_patterns=['*.safetensors', '*.json', '*.txt', '*.jinja'])
        return require_checkpoint(model)


def ray_checkpoint(directory):
    root = Path(directory)
    final = root / 'epoch-control/final-checkpoint.json'
    if final.exists():
        checkpoint = Path(read_json(final)['checkpoint'])
    else:
        step = int((root / 'checkpoints/latest_checkpointed_iteration.txt').read_text())
        checkpoint = root / 'checkpoints' / f'global_step_{step}'
    return require_checkpoint(checkpoint / 'actor/huggingface')


def launch(settings, module_args, log, env, distributed=False):
    if settings['mode'] == 'ray':
        from chartadept.ray_runtime import run_workers
        run_workers(settings, module_args, log, env, distributed)
    else:
        if distributed:
            module_args = ['torch.distributed.run', '--standalone', f"--nproc-per-node={settings['gpus_per_node']}",
                           '-m', *module_args]
        run_driver(module_args, log, env)


def run_driver(module_args, log, env):
    with log.open('w') as stream:
        result = subprocess.run([sys.executable, '-B', '-u', '-m', *module_args], cwd=ROOT,
                                env={**os.environ, **env}, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{module_args[0]} exited with status {result.returncode}. See {log}.')


def run_stage(stage, output, previous, data, args, settings, work):
    env = {'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'HF_HOME': os.environ['HF_HOME'],
           'PYTHONPATH': str(ROOT), 'PYTHONDONTWRITEBYTECODE': '1', 'TOKENIZERS_PARALLELISM': 'false'}
    for key in ('NCCL_SOCKET_IFNAME', 'GLOO_SOCKET_IFNAME'):
        if key in os.environ:
            env[key] = os.environ[key]
    mode = settings['mode']
    log = output.with_suffix('.log')
    print(f'Running {stage} using {mode}. Artifacts and logs under {output.parent}', flush=True)
    if stage == 'sft':
        model = model_checkpoint(work)
        output.mkdir(parents=True, exist_ok=True)
        for phase, folder in [('initial', 'initial'), ('continue', 'continuation')]:
            target = output / folder
            launch(settings, ['chartadept.train_sft', phase, '--model', str(model), '--output', str(target),
                   '--dataset-dir', str(data), '--epochs', str(args.epochs)], target.with_suffix('.log'), env,
                   distributed=True)
            model = require_checkpoint(target)
        return model

    if stage == 'gspo':
        options = ['--model', previous['sft'], '--data', str(data), '--output', str(output),
                   '--updates', str(args.updates)]
        if args.api_config:
            options.extend(['--api-config', str(args.api_config.resolve())])
        elif not read_json(data / 'ready.json')['numeric_rewards']:
            raise ValueError('Supply --api-config for semantic reward grading on nonnumeric training data.')
        if mode == 'local':
            launch(settings, ['chartadept.local_gspo', *options], log, env, distributed=True)
            return require_checkpoint(output / 'policy')
        if not args.api_config:
            options.append('--numeric-rewards')
        env.update(CHART_RL_RAY_ADDRESS=settings['address'], CHARTADEPT_PAGEABLE_BUFFERS='1')
        run_driver(['chartadept.train_rl', 'gspo', '--nodes', str(settings['nodes']),
                    '--gpus-per-node', str(settings['gpus_per_node']), *options], log, env)
        completed = read_json(output / 'epoch-control/coverage-status.json')['completed_step']
        if completed != args.updates:
            raise RuntimeError(f'GSPO completed {completed} of {args.updates} rollout updates in one pass. '
                               f'Use more training questions or request fewer updates. See {log}.')
        return ray_checkpoint(output)

    questions = data / 'questions.jsonl'
    module = 'chartadept.local_infer' if mode == 'local' else 'chartadept.infer'
    launch(settings, [module, '--model', previous['gspo'], '--questions', str(questions),
           '--output', str(output)], log, env)
    predictions = read_jsonl_by_id(output / 'index.jsonl')
    if predictions.keys() != read_jsonl_by_id(questions).keys() or any(
        p['status'] not in ('completed', 'budget_exhausted') for p in predictions.values()
    ):
        raise RuntimeError(f'Inference did not finish the question set. See {log}.')
    print(f'Generated {len(predictions)} predictions in {output / "index.jsonl"}.', flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', nargs='?', type=lambda s: 'infer' if s == 'inference' else s,
                        choices=STAGES, help='Omit to run all three stages.')
    parser.add_argument('--data', type=Path, help='SFT input directory with train, development and test JSONL files')
    parser.add_argument('--epochs', type=int, default=1, help='Epochs per SFT stage')
    parser.add_argument('--updates', type=int, default=3, help='GSPO rollout updates requested within one pass')
    parser.add_argument('--api-config', type=Path, help='GSPO semantic reward endpoint for nonnumeric training data')
    args = parser.parse_args()
    if args.epochs < 1 or args.updates < 1:
        parser.error('Epochs and updates must be positive.')
    if args.data and args.stage not in (None, 'sft'):
        parser.error('Select the dataset when starting SFT.')
    work = (ROOT.parent / os.environ.get('CHARTADEPT_WORK_DIR', '../chartadept-work')).resolve()
    settings_file = work / 'settings.json'
    if not settings_file.is_file():
        parser.error('Select a runtime with bash setup_env.sh as described in README.md.')
    settings = read_json(settings_file)
    mode = settings['mode']
    venv = work / f'venv-{mode}'
    if Path(sys.prefix).resolve() != venv.resolve():
        python = str(venv / 'bin/python')
        os.execv(python, [python, '-B', '-u', '-m', 'chartadept.run', *sys.argv[1:]])
    os.environ['PATH'] = str(venv / 'bin') + os.pathsep + os.environ.get('PATH', '')
    os.environ.setdefault('HF_HOME', str(work / 'hf-cache'))
    if mode == 'ray':
        os.environ.setdefault('RAY_USAGE_STATS_ENABLED', '0')
    else:
        import torch
        devices = 1 if args.stage == 'infer' else settings['gpus_per_node']
        if torch.cuda.device_count() < devices:
            parser.error(f'This stage needs {devices} visible local CUDA device(s).')

    topology = f"{mode}-{settings['nodes']}x{settings['gpus_per_node']}"
    runs = output_directory(os.environ.get('CHARTADEPT_RUN_DIR', work / 'runs')) / topology
    runs.mkdir(parents=True, exist_ok=True)
    state_file = runs / 'latest-run.json'
    with (runs / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for stage in ((args.stage,) if args.stage else STAGES):
            index = STAGES.index(stage)
            if stage == 'sft':
                run_dir = Path(tempfile.mkdtemp(prefix='run-', dir=runs))
                previous = {}
                from chartadept.experiment_data import prepare
                prepare(run_dir / 'data', args.data)
            else:
                prior_stage = STAGES[index - 1]
                state = read_json(state_file) if state_file.is_file() else None
                if not state or prior_stage not in state['outputs']:
                    parser.error(f'Run the {prior_stage} block first.')
                run_dir, previous = Path(state['directory']), state['outputs']
            output = run_dir / stage
            if output.exists() or output.with_suffix('.log').exists():
                output = Path(tempfile.mkdtemp(prefix=f'{stage}-', dir=run_dir))
            artifact = run_stage(stage, output, previous, run_dir / 'data', args, settings, work)
            outputs = {name: previous[name] for name in STAGES[:index]}
            outputs[stage] = str(artifact)
            write_json(state_file, {'directory': str(run_dir), 'outputs': outputs})
            print(f'Finished {stage}. Artifacts at {artifact}', flush=True)


if __name__ == '__main__':
    main()
