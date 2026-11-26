"""Replay saved tool calls on the example images; this does not run the model."""
import argparse
import json
from pathlib import Path
import sys

from chartadept.io import ROOT, read_jsonl_by_id, write_json
from chartadept.tools import Environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--questions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    questions = read_jsonl_by_id(args.questions)
    results = []
    for path in sorted((ROOT / 'examples/qualitative').glob('RQ-*.json')):
        case = json.loads(path.read_text())
        question = questions[case['id']]
        directory = args.output / case['id']
        environment = Environment([question['image']], directory, sys.executable)
        observations = []
        try:
            for call in case['tool_calls_recorded']:
                observation = environment.execute(call['name'], call['arguments'])
                if observation['status'] != 'success' or observation.get('stdout', '') != call['stdout']:
                    raise ValueError(f'Tool replay differs for {case["id"]}')
                observations.append(observation)
        finally:
            environment.close()
        results.append({'id': case['id'], 'mode': case['mode'],
            'tool_observations': observations, 'saved_answer': case['final_answer']})
    write_json(args.output / 'replay.json', {'operation': 'recorded-tool replay, not fresh model inference', 'cases': results})
    print(args.output / 'replay.json')


if __name__ == '__main__':
    main()
