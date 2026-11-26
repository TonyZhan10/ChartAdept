"""Materialize only the final RL inputs and their packaged chart images."""
import argparse
import tarfile
from pathlib import Path

from chartadept.io import ROOT, output_directory, write_json


def prepare_rl(output):
    import pyarrow as pa
    import pyarrow.parquet as pq
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    marker = output / 'ready.json'
    if marker.exists():
        return output
    with tarfile.open(ROOT / 'data/rl/images.tar') as archive:
        archive.extractall(output, filter='data')
    counts = {}
    for split in ['train', 'development']:
        table = pq.read_table(ROOT / f'data/rl/{split}.parquet')
        rows = table.to_pylist()
        for row in rows:
            for image in row['images']:
                image['image'] = str(output / image['image'])
            row['extra_info']['image_path'] = row['images'][0]['image']
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), output / f'{split}.parquet')
        counts[split] = len(rows)
    write_json(marker, counts)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=output_directory, required=True)
    args = parser.parse_args()
    print(prepare_rl(args.output))
