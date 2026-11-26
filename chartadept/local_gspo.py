"""Single-host, data-parallel GSPO with Transformers rollouts and PyTorch collectives."""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
from math import gcd
import os
from pathlib import Path
import random

import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from chartadept.api import chat_payload, request_json
from chartadept.chat import parsed_json
from chartadept.decimal_precision import parse_number
from chartadept.gspo.selection import select_groups
from chartadept.io import ROOT, append_json, output_directory, read_json, write_json
from chartadept.local_policy import initial_context, load_policy, rollout, vision_input_token_ids
from chartadept.reward.broker import RUBRIC, terminal_rule


RECIPE = read_json(ROOT / 'configs/gspo-recipe.json')
MINIBATCHES = RECIPE['optimizer_steps_per_rollout_update']


def grade(answer, row, judge, config_path):
    extra = row['extra_info']
    reference = row['reward_model']['ground_truth']
    answer_spec = extra.get('answer_spec')
    correct, source = terminal_rule(answer, reference, extra['question'], answer_spec)
    if correct is not None:
        return float(correct), source
    if judge is None:
        return 0.0, 'numeric_answer_required'
    values = {'question': extra['question'], 'reference_answer': reference, 'model_final_answer': answer}
    if answer_spec:
        values['verified_answer_specification'] = answer_spec
    payload = chat_payload(RUBRIC + '\nACTUAL GRADING INPUT:\n' + json.dumps(values, ensure_ascii=False),
                           config=judge, max_tokens=1024)
    response = request_json('POST', '/v1/chat/completions', payload, config_path=config_path,
                            timeout=judge.get('timeout', 600), retries=judge.get('retries', 0))
    score = parsed_json(response['choices'][0]['message']['content'])['score']
    if type(score) is not int or score not in (0, 1):
        raise ValueError('The semantic judge must return an integer score of 0 or 1')
    return float(score), judge['model']


def action_logprobs(actor, trajectory, device):
    inputs = {key: value.to(device) for key, value in trajectory.inputs.items()}
    positions = torch.tensor(trajectory.action_positions, device=device)
    targets = inputs['input_ids'][0, positions]
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        logits = actor(**inputs, use_cache=False, logits_to_keep=positions - 1).logits[0]
    policy = actor.module if isinstance(actor, DistributedDataParallel) else actor
    logits = logits.float()
    logits[:, vision_input_token_ids(policy.config)] = -torch.inf
    return -F.cross_entropy(logits, targets, reduction='none')


def gspo_loss(current, old, advantage):
    # Each vector contains every sampled assistant token in this interaction, across all turns.
    ratio = (current - old).mean().clamp(max=10).exp()
    clipped = ratio.clamp(1 - RECIPE['clip_ratio_low'], 1 + RECIPE['clip_ratio_high'])
    return -torch.minimum(ratio * advantage, clipped * advantage), ratio.detach()


def update_policy(policy, actor, optimizer, selected, device, world):
    # Finish scoring every selected trajectory with the rollout policy before changing weights.
    with torch.no_grad():
        old = [action_logprobs(policy, trajectory, device) for trajectory, _ in selected]
    actor.train()
    size = len(selected) // MINIBATCHES
    losses, ratios, norms = [], [], []
    for start in range(0, len(selected), size):
        optimizer.zero_grad(set_to_none=True)
        for index in range(start, start + size):
            trajectory, advantage = selected[index]
            sync = actor.no_sync() if world > 1 and index < start + size - 1 else nullcontext()
            with sync:
                current = action_logprobs(actor, trajectory, device)
                loss, ratio = gspo_loss(current, old[index], advantage)
                (loss / size).backward()
            losses.append(loss.detach())
            ratios.append(ratio)
        norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0, error_if_nonfinite=True)
        norms.append(norm.detach())
        optimizer.step()
    policy.eval()
    metrics = torch.stack([torch.stack(losses).mean(), (torch.stack(ratios) - 1).abs().mean(),
                           torch.stack(norms).mean()])
    if world > 1:
        dist.all_reduce(metrics)
        metrics /= world
    return dict(zip(('loss', 'mean_abs_sequence_ratio_change', 'mean_preclip_grad_norm'), metrics.tolist()))


def sample_group(policy, processor, row, ordinal, update, args, rank, world, device, judge):
    extra = row['extra_info']
    qid = extra['question_id']
    source = (args.data / extra['image_path']).resolve()
    prompt = row['prompt'][-1]['content'].removeprefix('<image>').lstrip('\n')
    initial = initial_context(processor, source, prompt, row['prompt'][0]['content'])
    k = args.rollouts_per_prompt
    rewards = torch.zeros(k, device=device)
    local = []
    for sample in range(rank, k, world):
        seed = args.seed + ordinal * k + sample
        torch.manual_seed(seed)
        directory = args.output / 'rollouts' / f'question-{ordinal:05d}' / f'sample-{sample:02d}'
        trajectory = rollout(policy, processor, initial, source, directory / 'tool-work', sample=True,
                             max_tokens=args.max_tokens, context_limit=args.context_length)
        score, grading_source = grade(trajectory.trace['final_answer'], row, judge, args.api_config)
        trajectory.trace.update(id=qid, sample=sample, seed=seed, source_policy_update=update - 1,
                                reward=score, grading_source=grading_source)
        write_json(directory / 'trace.json', trajectory.trace)
        rewards[sample] = score
        local.append((sample, trajectory))
    if world > 1:
        dist.all_reduce(rewards)
    scores = rewards.tolist()
    selection = select_groups([qid] * k, [{'score': s} for s in scores], k, RECIPE['selection_epsilon'])
    accepted = selection['groups'][0]['kept']
    if rank == 0:
        append_json(args.output / 'groups.jsonl', {'update': update, 'id': qid, 'rewards': scores, 'kept': accepted})
    if not accepted:
        return []
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
    return [(trajectory, advantages[sample]) for sample, trajectory in local]


def train(args, rank, world, device):
    rows = pq.read_table(args.data / 'train.parquet').to_pylist()
    judge = read_json(args.api_config) if args.api_config else None
    if judge is None and any(parse_number(row['reward_model']['ground_truth']) is None for row in rows):
        raise ValueError('Supply --api-config to grade nonnumeric training answers')
    random.Random(args.seed).shuffle(rows)
    policy, processor = load_policy(args.model, device, training=True)
    policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    if world > 1:
        from torch.distributed.optim import ZeroRedundancyOptimizer
        actor = DistributedDataParallel(policy, device_ids=[device.index] if device.type == 'cuda' else None,
                                        gradient_as_bucket_view=True, broadcast_buffers=False)
        optimizer = ZeroRedundancyOptimizer(policy.parameters(), optimizer_class=torch.optim.AdamW,
                                            lr=args.learning_rate, weight_decay=RECIPE['weight_decay'],
                                            fused=device.type == 'cuda')
    else:
        actor = policy
        optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=RECIPE['weight_decay'],
                                      fused=device.type == 'cuda')
    if rank == 0:
        write_json(args.output / 'recipe.json', {'backend': 'torchrun-transformers', 'world_size': world,
                   'initial_model': str(args.model.resolve()), 'seed': args.seed, 'updates': args.updates,
                   'rollouts_per_prompt': args.rollouts_per_prompt, 'groups_per_update': args.groups_per_update,
                   'optimizer_steps_per_rollout_update': MINIBATCHES, 'learning_rate': args.learning_rate,
                   'clip_ratio_low': RECIPE['clip_ratio_low'], 'clip_ratio_high': RECIPE['clip_ratio_high'],
                   'max_generated_tokens': args.max_tokens, 'context_length': args.context_length,
                   'reward': 'binary_terminal_correctness', 'semantic_judge': judge['model'] if judge else None})
    cursor = 0
    for update in range(1, args.updates + 1):
        selected = []
        start = cursor
        local_batch = args.groups_per_update * args.rollouts_per_prompt // world
        while len(selected) < local_batch and cursor < len(rows):
            selected.extend(sample_group(policy, processor, rows[cursor], cursor, update, args, rank, world, device, judge))
            cursor += 1
        # At the end of the one-pass pool, retain complete K-groups and four equal minibatches.
        local_group = args.rollouts_per_prompt // world
        quantum = local_group * MINIBATCHES // gcd(local_group, MINIBATCHES)
        selected = selected[:len(selected) // quantum * quantum]
        if not selected:
            raise RuntimeError(f'Only {update - 1} of {args.updates} updates completed in one question pass. '
                               'Use more questions with mixed rollout rewards or request fewer updates.')
        values = update_policy(policy, actor, optimizer, selected, device, world)
        if rank == 0:
            report = {'update': update, 'optimizer_steps': MINIBATCHES, 'optimizer_steps_total': update * MINIBATCHES,
                      'candidate_groups': cursor - start, 'visited_questions': cursor,
                      'selected_groups': len(selected) * world // args.rollouts_per_prompt,
                      'selected_trajectories': len(selected) * world, **values}
            append_json(args.output / 'metrics.jsonl', report)
            print(json.dumps(report), flush=True)
    if rank == 0:
        export = args.output / 'policy'
        policy.save_pretrained(export)
        processor.save_pretrained(export)
        print(f'Exported the GSPO policy to {export}', flush=True)
    if world > 1:
        dist.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True, help='Prepared directory containing train.parquet')
    parser.add_argument('--output', type=output_directory, required=True)
    parser.add_argument('--updates', type=int, default=3)
    parser.add_argument('--rollouts-per-prompt', type=int, default=8)
    parser.add_argument('--groups-per-update', type=int, default=4)
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--context-length', type=int, default=8192)
    parser.add_argument('--learning-rate', type=float, default=RECIPE['learning_rate'])
    parser.add_argument('--seed', type=int, default=RECIPE['seed'])
    parser.add_argument('--api-config', type=Path, help='Semantic judge configuration for nonnumeric answers')
    args = parser.parse_args()
    world, rank = int(os.environ.get('WORLD_SIZE', 1)), int(os.environ.get('RANK', 0))
    if min(args.updates, args.groups_per_update, args.max_tokens, args.context_length, args.learning_rate) <= 0:
        parser.error('Updates, group count, token limits, and learning rate must be positive')
    if args.rollouts_per_prompt < 2 or args.rollouts_per_prompt % world or (
            args.rollouts_per_prompt * args.groups_per_update) % (world * MINIBATCHES):
        parser.error('K must be divisible by the rank count and K × groups by ranks × four minibatches')
    if (args.output / 'recipe.json').exists() or (args.output / 'policy').exists():
        parser.error('This output already contains a GSPO run. Use a fresh output directory.')
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo', timeout=timedelta(hours=1))
    try:
        train(args, rank, world, device)
    finally:
        if world > 1:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
