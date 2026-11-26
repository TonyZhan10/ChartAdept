"""Paired figure-cluster bootstrap for accuracy differences."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from chartadept.io import read_jsonl_by_id, write_json

BENCHMARK = {'CharXiv-RQ': 'CharXiv', 'CharXiv-DQ': 'CharXiv', 'ChartQA': 'ChartQA',
    'ChartQAPro': 'ChartQAPro', 'ChartBench-NQA': 'ChartBench', 'ChartBench-Binary': 'ChartBench',
    'ChartX': 'ChartX', 'ReachQA-Reas': 'ReachQA', 'ReachQA-Recog': 'ReachQA', 'ChartMuseum': 'ChartMuseum'}


def paired_bootstrap(questions, grades, baseline, samples=20000, seed=1729):
    metrics = sorted({q['metric'] for q in questions.values()})
    position = {metric: index for index, metric in enumerate(metrics)}
    groups = defaultdict(dict)
    for qid, q in questions.items():
        benchmark = q.get('benchmark') or BENCHMARK[q['metric']]
        key = str(q['figure_id'])
        sums, counts = groups[benchmark].setdefault(key,
            (np.zeros(len(metrics)), np.zeros(len(metrics))))
        col = position[q['metric']]
        sums[col] += int(grades.get(qid, {}).get('judge_label') == 'A') - int(baseline.get(qid, {}).get('judge_label') == 'A')
        counts[col] += 1
    strata = [(np.array([v[0] for v in group.values()]), np.array([v[1] for v in group.values()]))
              for group in groups.values()]
    total = sum(x.sum(axis=0) for x, _ in strata)
    count = sum(n.sum(axis=0) for _, n in strata)
    observed = 100 * total / count
    draws = np.empty((samples, len(metrics)))
    rng = np.random.default_rng(seed)
    for iteration in range(samples):
        total, count = np.zeros(len(metrics)), np.zeros(len(metrics))
        for sums, counts in strata:
            indices = rng.integers(0, len(sums), size=len(sums))
            total += sums[indices].sum(axis=0)
            count += counts[indices].sum(axis=0)
        draws[iteration] = np.divide(100 * total, count,
            out=np.full(len(metrics), np.nan), where=count > 0)
    result = {'samples': samples, 'seed': seed, 'unit': 'percentage points',
              'resampling_unit': 'figure within benchmark; all questions and both models kept together',
              'metrics': {}}
    for metric, col in position.items():
        result['metrics'][metric] = {'difference_pp': float(observed[col]),
            'percentile_95_ci_pp': np.nanquantile(draws[:, col], [.025, .975]).tolist()}
    result['macro_average'] = {'difference_pp': float(observed.mean()),
        'percentile_95_ci_pp': np.nanquantile(draws.mean(axis=1), [.025, .975]).tolist()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--questions', type=Path, required=True)
    parser.add_argument('--grades', type=Path, required=True)
    parser.add_argument('--baseline-grades', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=1729)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('samples must be positive')
    result = paired_bootstrap(read_jsonl_by_id(args.questions), read_jsonl_by_id(args.grades),
        read_jsonl_by_id(args.baseline_grades), args.samples, args.seed)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
