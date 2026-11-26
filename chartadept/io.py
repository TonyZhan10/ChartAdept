"""Read and write run artifacts."""
from argparse import ArgumentTypeError
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT.parent / 'chartadept-runs'


def output_directory(value):
    directory = Path(value).resolve()
    if directory.is_relative_to(ROOT):
        raise ArgumentTypeError(f'Choose an output directory outside the code directory {ROOT}')
    return directory


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def append_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl_by_id(path):
    with Path(path).open() as stream:
        return {row['id']: row for row in map(json.loads, stream)}
