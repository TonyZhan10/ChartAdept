# Evaluation

## Questions and benchmark scope

Obtain benchmark images and annotations from their respective releases. Pass a JSONL file with one official evaluation item per line:

```json
{"id":"RQ-1969","image":"/absolute/path/1969.jpg","question":"Official question","prompt":"Official question and benchmark instructions","reference_answer":"Official reference","task":"RQ","metric":"CharXiv-RQ","benchmark":"CharXiv","figure_id":"1969"}
```

`prompt` preserves the benchmark's answer-format instructions. The inference loop sends the image and prompt to the answering model; `reference_answer` is used only by the separate evaluator. For conversational ChartQAPro items, preserve the conversation in the prompt and use the final reference answer. Use absolute image paths for portable manifests.

| Benchmark split | Metric | Questions |
|---|---|---:|
| CharXiv validation | CharXiv-RQ / CharXiv-DQ | 1,000 / 4,000 |
| ChartQA test, human + augmented | ChartQA | 2,500 |
| ChartQAPro test | ChartQAPro | 1,948 |
| ChartBench test | ChartBench-NQA / ChartBench-Binary | 2,100 / 16,800 |
| ChartX test, QA | ChartX | 1,152 |
| ReachQA test | ReachQA-Reas / ReachQA-Recog | 1,000 / 1,000 |
| ChartMuseum test | ChartMuseum | 1,000 |

The complete scope contains 32,500 questions and ten accuracy metrics. The paper Avg. is the equal-weight mean of those ten metrics. A run on a subset should be identified by its actual scope.

## Generation

```bash
python -m chartadept.infer --server http://127.0.0.1:8237 \
  --questions /path/to/questions.jsonl --preset main \
  --output ../chartadept-runs/evaluation/main
```

Run Museum separately with `--preset museum`. The main preset uses 16K greedy generation. Museum uses a 32K budget and temperature 0.7, with up to 4K reserved inside the total for a tool-free JSON answer if a limit is reached or the answer cannot be parsed. The `budget-control-32k` preset uses the same budget and reserve at temperature zero for the fixed-weight controls. `--mode direct` disables tools.

The main agent interface uses a 65,536-token context, image pixel limits 802,816--3,211,264, at most twelve assistant turns, and six tool calls. Preserve the supplied inference configurations when comparing a model against the reported protocol.

## Full-response Compass grading

Serve CompassVerifier-7B through an OpenAI-compatible endpoint using its native 32K context. Obtain its tokenizer separately. The upstream ChartVerse rubric is fetched as a separate third-party input:

```bash
python scripts/fetch_compass_rubric.py \
  --output ../chartadept-runs/compass-rubric.json

python -m chartadept.evaluate \
  --questions /path/to/questions.jsonl \
  --predictions ../chartadept-runs/evaluation/main/index.jsonl \
  --grades ../chartadept-runs/evaluation/main-grades/grades.jsonl \
  --output ../chartadept-runs/evaluation/main-grades/summary.json \
  --judge-base-url http://127.0.0.1:8765/v1 \
  --judge-model compass-verifier-7b \
  --judge-tokenizer /path/to/CompassVerifier-7B \
  --rubric ../chartadept-runs/compass-rubric.json
```

Set `COMPASS_API_KEY` in the process environment if the judge requires authentication. The answering-model service and judge service are separate. Retain the exact rubric JSON with each evaluation run; `--source-file` can extract it from a saved upstream implementation.

The evaluator reads the complete last assistant response from the saved trace, including separately returned reasoning when present. It does not score only the extracted answer tag. It accounts for rubric, question, reference, chat-template tokens, and a 2,048-token judge output reserve. Only overlong responses are shortened, by removing a prefix to fit the native 32K context.

Explicit A/B/C judgments produce scores 1/0/0. The default permits six attempts for request or verdict failures; unresolved judgments and missing predictions contribute zero to the official question denominator and remain explicitly marked unresolved. Request payloads and raw judge responses are retained in the run directory. Use a new grades directory for a different checkpoint, prediction set, rubric, or protocol.

To aggregate an existing grades file without a judge:

```bash
python -m chartadept.evaluate --summarize-only \
  --questions /path/to/questions.jsonl \
  --grades /path/to/grades.jsonl --output ../chartadept-runs/summary.json
```

## Paired intervals and tables

For paired comparisons, each question must have a canonical `figure_id` and `benchmark`. Questions from the same figure share that ID even if image filenames differ. Keep official source identifiers when preparing data; per-question filenames do not define independent figure groups.

```bash
python -m chartadept.statistics \
  --questions /path/to/questions.jsonl \
  --grades /path/to/chartadept-grades.jsonl \
  --baseline-grades /path/to/baseline-grades.jsonl \
  --samples 20000 --seed 1729 --output ../chartadept-runs/paired-intervals.json

python scripts/export_results.py --output ../chartadept-runs/reported-table.csv
```

Bootstrap sampling is within benchmark and jointly preserves all questions, metrics, and both model outputs for each sampled figure. Macro differences use the equal-weight mean of included metrics. The CSV exporter transcribes the supplied reported results; it does not regenerate predictions or recompute those published scores. Complete final per-question files for Table 2 are not bundled in this release; the three qualitative records are saved individual examples.
