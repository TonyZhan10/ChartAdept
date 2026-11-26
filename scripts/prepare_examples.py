"""Prepare the three held-out examples from the bundled or official CharXiv images."""
import argparse
import json
from pathlib import Path
import shutil

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--charxiv-images', type=Path,
                        help='Optional official CharXiv images directory; defaults to the three bundled examples.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    records = root / 'examples/qualitative'
    sources = json.loads((root / 'docs/assets/sources.json').read_text())
    questions = []
    for path in sorted(records.glob('RQ-*.json')):
        case = json.loads(path.read_text())
        source = (args.charxiv_images / f'{case["official_id"]}.jpg' if args.charxiv_images
                  else root / 'docs/assets' / sources[case['id']]['input'])
        with Image.open(source) as image:
            if list(image.size) != case['input_size']:
                raise ValueError(f'{source}: expected the evaluation image size {case["input_size"]}, got {image.size}')
        destination = args.output / source.name
        shutil.copyfile(source, destination)
        questions.append({'id': case['id'], 'image': str(destination.resolve()),
            'question': case['question'], 'prompt': case['prompt'],
            'reference_answer': case['reference_answer'], 'task': 'RQ',
            'metric': 'CharXiv-RQ', 'benchmark': 'CharXiv', 'figure_id': case['official_id']})
    with (args.output / 'questions.jsonl').open('w') as stream:
        for question in questions:
            stream.write(json.dumps(question, ensure_ascii=False) + '\n')
    print(args.output / 'questions.jsonl')


if __name__ == '__main__':
    main()
