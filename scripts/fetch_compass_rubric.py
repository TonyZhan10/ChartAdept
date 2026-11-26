"""Read the two public Compass rubric constants without executing upstream code."""
import argparse
import ast
import json
from pathlib import Path
from urllib.request import urlopen

SOURCE = 'https://raw.githubusercontent.com/starriver030515/ChartVerse/main/eval/verify.py'


def extract(source):
    names = {'SYSTEM_PROMPT': 'system_prompt', 'CV_COT_PROMPT': 'user_template'}
    values = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    values[names[target.id]] = ast.literal_eval(node.value)
    if set(values) != set(names.values()):
        raise ValueError('Source must define SYSTEM_PROMPT and CV_COT_PROMPT.')
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-file', type=Path, help='Use a saved upstream source instead of downloading.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.source_file:
        source = args.source_file.read_text()
    else:
        with urlopen(SOURCE, timeout=60) as response:
            source = response.read().decode('utf-8')
    result = {**extract(source), 'upstream_source': SOURCE}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
