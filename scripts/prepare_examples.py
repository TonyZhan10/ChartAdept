"""Prepare the three held-out examples using separately obtained CharXiv images."""
import argparse
import json
from pathlib import Path
import shutil

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--charxiv-images', type=Path, required=True,
                        help='Official CharXiv images directory containing 1969.jpg, 2201.jpg, and 913.jpg.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = Path(__file__).resolve().parents[1] / 'examples/qualitative'
    questions = []
    for path in sorted(records.glob('RQ-*.json')):
        case = json.loads(path.read_text())
        source = args.charxiv_images / f'{case["official_id"]}.jpg'
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
