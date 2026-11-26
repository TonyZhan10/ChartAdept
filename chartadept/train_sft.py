"""Train the initial or continuation SFT stage on chart demonstrations."""
import argparse
import json
import os
from pathlib import Path

from chartadept.io import ROOT, output_directory, read_json, write_json


def settings(stage, output=None, resume=None, model=None):
    import yaml
    config = yaml.safe_load((ROOT / f'configs/sft-{stage}.yaml').read_text())
    for key in ('model_name_or_path', 'dataset_dir', 'tokenized_path', 'deepspeed'):
        config[key] = str((ROOT / config[key]).resolve())
    config['output_dir'] = str(output_directory(output or ROOT / config['output_dir']))
    if model:
        config['model_name_or_path'] = str(Path(model).resolve())
    if resume:
        config['resume_from_checkpoint'] = str(Path(resume).resolve())
        zero = read_json(config['deepspeed'])
        zero['checkpoint']['load_universal'] = False
        config['deepspeed'] = zero
    return config


def dataset_settings(config, directory, epochs):
    """Train a native SFT stage on the supplied chart demonstrations."""
    directory = directory.resolve()
    config.pop('tokenized_path')
    config.update(dataset='chart_train', eval_dataset='chart_development', dataset_dir=str(directory),
        media_dir=str(directory), data_shared_file_system=True, deepspeed=None, pure_bf16=True, cutoff_len=4096,
        per_device_train_batch_size=2, num_train_epochs=epochs, eval_strategy='epoch',
        prediction_loss_only=True, save_strategy='no', preprocessing_num_workers=1, dataloader_num_workers=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['initial', 'continue'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--model', type=Path, help='Override the base or preceding SFT checkpoint')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--dataset-dir', type=Path, help='Raw LLaMA-Factory chart_train and chart_development datasets')
    parser.add_argument('--epochs', type=int, default=1, help='Epochs per stage with --dataset-dir')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    try:
        config = settings(args.stage, args.output, args.resume, args.model)
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    world = 128 if args.stage == 'initial' else 160
    if args.dataset_dir:
        dataset_settings(config, args.dataset_dir, args.epochs)
        world = int(os.environ.get('WORLD_SIZE', '1'))
    if args.dry_run:
        print(json.dumps({'world_size': world, 'config': config}, indent=2))
        return
    if int(os.environ.get('WORLD_SIZE', '1')) != world:
        parser.error(f'This saved stage requires WORLD_SIZE={world}; use torchrun as documented')
    from chartadept.checkpoints import require_checkpoint
    try:
        require_checkpoint(config.get('resume_from_checkpoint') or config['model_name_or_path'])
    except ValueError as error:
        parser.error(str(error))
    schedule = ROOT / f'configs/sampling-{args.stage}.json'
    if not args.dataset_dir:
        if not (Path(config['tokenized_path']) / 'dataset_dict.json').is_file():
            parser.error('Place the prepared SFT DatasetDict at data/sft before starting a full stage')
        if not schedule.is_file():
            parser.error(f'Place the saved SFT route schedule at {schedule.relative_to(ROOT)}')
    os.chdir(ROOT)
    from llamafactory.train.tuner import run_exp
    callbacks = []
    if not args.dataset_dir:
        from chartadept.route_sampler import install
        from transformers import TrainerCallback

        sampler, _ = install(schedule, 90474)

        class EpochCallback(TrainerCallback):
            def on_epoch_begin(self, arguments, state, control, **kwargs):
                sampler.set_epoch(int(state.epoch or 0))

        callbacks.append(EpochCallback())
    if int(os.environ.get('RANK', '0')) == 0:
        write_json(Path(config['output_dir']) / 'effective-config.json', config)
    run_exp(config, callbacks=callbacks)


if __name__ == '__main__':
    main()
