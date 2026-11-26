"""Export the reported main-comparison table as CSV without recomputing scores."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('results/reported-main-table.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ['Model', *data['metric_order'], 'Avg10']
    with args.output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for model in data['models']:
            writer.writerow({'Model': model['name'], **model['accuracy_percent'], 'Avg10': model['Avg10']})
    print(args.output)


if __name__ == '__main__':
    main()
