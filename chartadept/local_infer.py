"""Generate chart answers directly with a local Transformers checkpoint."""
import argparse
import json
from pathlib import Path

import torch

from chartadept.io import output_directory, write_json
from chartadept.local_policy import PROTOCOL, initial_context, load_policy, rollout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--questions', required=True, type=Path)
    parser.add_argument('--output', required=True, type=output_directory)
    parser.add_argument('--max-tokens', type=int, default=PROTOCOL['total_completion_token_budget'])
    args = parser.parse_args()
    if args.max_tokens < 1:
        parser.error('--max-tokens must be positive')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, processor = load_policy(args.model, device, training=False)
    args.output.mkdir(parents=True, exist_ok=True)
    pending_index = args.output / 'index.jsonl.tmp'
    with args.questions.open() as questions, pending_index.open('w') as index:
        for number, question in enumerate(map(json.loads, questions), 1):
            source = (args.questions.parent / question['image']).resolve()
            prompt = question.get('prompt', question['question'])
            inputs = initial_context(processor, source, prompt)
            directory = args.output / 'traces' / f'question-{number:05d}'
            result = rollout(model, processor, inputs, source, directory / 'tool-work', sample=False,
                             max_tokens=args.max_tokens, context_limit=PROTOCOL['max_model_len'])
            trace_file = directory / 'trace.json'
            result.trace.update(id=question['id'], method='zoom_python', task=question.get('task', 'chart-qa'),
                                question=question['question'], image=str(source), model=str(args.model.resolve()),
                                trace_file=str(trace_file.relative_to(args.output)))
            write_json(trace_file, result.trace)
            record = {key: value for key, value in result.trace.items() if key not in ('turns', 'tools')}
            print(json.dumps(record, ensure_ascii=False), file=index, flush=True)
            print(f"{record['id']}: {record['termination']} ({record['tool_calls']} tool calls)", flush=True)
    pending_index.replace(args.output / 'index.jsonl')


if __name__ == '__main__':
    main()
