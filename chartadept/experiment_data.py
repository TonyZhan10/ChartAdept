"""Prepare chart questions for SFT, GSPO, and held-out inference."""
import json
from io import BytesIO
from pathlib import Path

from chartadept.decimal_precision import parse_number
from chartadept.io import ROOT, read_json, write_json


def public_split(split, count, images, used_images):
    from datasets import Image, load_dataset
    from PIL import Image as PILImage

    data = load_dataset('HuggingFaceM4/ChartQA', split=split, streaming=True, token=False)
    data = data.cast_column('image', Image(decode=False))
    rows = []
    for index, item in enumerate(data):
        labels = item['label']
        if len(labels) != 1 or parse_number(labels[0]) is None:
            continue
        pixels = item['image']['bytes']
        if pixels in used_images:
            continue
        used_images.append(pixels)
        qid = f'chartqa-{split}-{index}'
        path = images / f'{qid}.png'
        with PILImage.open(BytesIO(pixels)) as chart:
            chart.convert('RGB').save(path)
        question = item['query']
        rows.append({'id': qid, 'image': str(path), 'question': question,
                     'prompt': question + '\nGive the numerical answer in <answer>...</answer>.',
                     'reference_answer': str(labels[0]), 'task': 'ChartQA'})
        if len(rows) == count:
            return rows
    raise ValueError(f'ChartQA {split} has fewer than {count} eligible questions')


def read_split(source, split):
    rows = []
    with (source / f'{split}.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            image = (source / row['image']).resolve()
            if not image.is_file():
                raise FileNotFoundError(image)
            row.update(id=str(row['id']), image=str(image))
            if split != 'test':
                row['reference_answer'] = str(row['reference_answer'])
            row.setdefault('prompt', row['question'])
            row.setdefault('task', 'chart-qa')
            rows.append(row)
    return rows


def prepare(output, source=None):
    import pyarrow as pa
    import pyarrow.parquet as pq

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if source is None:
        images = output / 'images'
        images.mkdir()
        used_images = []
        training = public_split('train', 320, images, used_images)
        splits = {'train': training[:64], 'development': public_split('val', 8, images, used_images),
                  'test': public_split('test', 8, images, used_images)}
        gspo_rows = training[64:]
    else:
        splits = {name: read_split(source.resolve(), name) for name in ('train', 'development', 'test')}
        gspo_rows = read_split(source.resolve(), 'gspo') if (source / 'gspo.jsonl').is_file() else splits['train']
    for training_rows in (splits['train'], gspo_rows):
        partitions = (training_rows, splits['development'], splits['test'])
        ids = [row['id'] for rows in partitions for row in rows]
        if not all(partitions) or len(set(ids)) != len(ids):
            raise ValueError('Provide nonempty splits with unique question IDs.')

    system = read_json(ROOT / 'configs/inference.json')['system_prompts']['zoom_python']
    def prompt_for(row):
        return [{'role': 'system', 'content': system}, {'role': 'user', 'content': '<image>\n' + row['prompt']}]

    datasets = {}
    for split in ('train', 'development'):
        examples = []
        for row in splits[split]:
            answer = {'role': 'assistant', 'content': '<answer>' + row['reference_answer'] + '</answer>'}
            examples.append({'images': [row['image']], 'messages': row.get('messages', [*prompt_for(row), answer])})
        questions = []
        for row in (gspo_rows if split == 'train' else splits[split]):
            questions.append({
                'data_source': row['task'], 'prompt': prompt_for(row), 'images': [{'image': row['image']}],
                'reward_model': {'style': 'rule', 'ground_truth': row['reference_answer']},
                'extra_info': {'question_id': row['id'], 'image_path': row['image'],
                               'question': row['question'], 'route_intervention': 'adaptive'}})
        filename = f'sft-{split}.json'
        write_json(output / filename, examples)
        datasets[f'chart_{split}'] = {'file_name': filename, 'formatting': 'sharegpt',
            'columns': {'messages': 'messages', 'images': 'images'},
            'tags': {'role_tag': 'role', 'content_tag': 'content', 'system_tag': 'system',
                     'user_tag': 'user', 'assistant_tag': 'assistant',
                     'observation_tag': 'observation', 'function_tag': 'function_call'}}
        pq.write_table(pa.Table.from_pylist(questions), output / f'{split}.parquet')
    write_json(output / 'dataset_info.json', datasets)
    with (output / 'questions.jsonl').open('w') as stream:
        for row in splits['test']:
            record = {k: row[k] for k in ('id', 'image', 'question', 'prompt', 'task')}
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    counts = {'sft_train': len(splits['train']), 'train': len(gspo_rows),
              'development': len(splits['development']), 'test': len(splits['test'])}
    write_json(output / 'ready.json', {**counts, 'numeric_rewards': all(
        parse_number(r['reference_answer']) is not None for rows in (gspo_rows, splits['development']) for r in rows)})
    print(f"Prepared {counts['sft_train']} SFT, {counts['train']} GSPO, {counts['development']} development "
          f"and {counts['test']} test questions.", flush=True)
