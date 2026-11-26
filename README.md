# ChartAdept

**ChartAdept: Learning to Answer and Act with Verified Chart Trajectories**

ChartAdept answers scientific-chart questions directly or obtains additional visual evidence with Zoom and Python. This repository provides training and inference implementations, the tool runtime, evaluation code, reported results, and three recorded qualitative examples.

The paper's training corpus and its data construction and verification pipeline are not included in this release. Training entry points accept user-provided data. Training-reward scoring and benchmark evaluation are included.

## Model

The final model is the GSPO step-162 export based on `Qwen/Qwen3-VL-4B-Instruct`. The checkpoint includes all four weight shards, tokenizer, image processor, and chat template. The original float32 export is approximately 19.3 GB; inference loads it in BF16. The public weight download location will be added when the model repository is published.

For a local checkout with the supplied checkpoint folder, use `../chartadept-model` in the commands below.

## Installation and inference

Use Linux, Python 3.12, and a CUDA-capable NVIDIA GPU. The reference runtime uses CUDA 12.8 and 80 GB GPUs. Memory use depends on image size, generation length, and concurrency.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-inference.txt
pip install -e . --no-deps

python -m chartadept.serve --model ../chartadept-model --port 8237 --max-sequences 1
```

In another terminal using the same environment:

```bash
python -m chartadept.infer \
  --server http://127.0.0.1:8237 \
  --image /absolute/path/to/chart.png \
  --question "What is the difference between the two values?" \
  --output ../chartadept-runs/example
```

The output contains final predictions, model turns, tool arguments, and actual tool observations. `--mode direct` disables tools for a controlled comparison. The default autonomous mode can itself answer without a tool call.

| Preset | Output budget | Temperature | Final-answer reserve |
|---|---:|---:|---|
| `main` | 16,384 tokens across all turns | 0 | None |
| `museum` | 32,768 tokens across all turns | 0.7 | Up to 4,096 tokens within the total budget |
| `budget-control-32k` | 32,768 tokens across all turns | 0 | Up to 4,096 tokens within the total budget |

Use `--preset museum` for the main ChartMuseum setting. All agent presets permit at most twelve assistant turns and six tool calls. Python tool calls execute code in a child process; use an isolated working environment for model-generated code.

## Examples

The saved examples are held-out benchmark cases, not training samples:

| Example | Observed behavior | Saved answer |
|---|---|---|
| CharXiv RQ-1969 | Direct answer, no tool calls | Negatively correlated |
| CharXiv RQ-2201 | Two Zoom calls linking Power and FDR | 0.025 |
| CharXiv RQ-913 | Two Zoom calls followed by Python | 1.5 |

[`examples/qualitative`](examples/qualitative) contains the original question, saved responses, actual tool arguments, and stdout. Obtain the benchmark images separately from [CharXiv](https://github.com/princeton-nlp/CharXiv).

```bash
python scripts/prepare_examples.py --charxiv-images /path/to/CharXiv/images \
  --output ../chartadept-runs/examples

# Replay the recorded tools without calling a model.
python -m chartadept.replay_examples \
  --questions ../chartadept-runs/examples/questions.jsonl \
  --output ../chartadept-runs/replay

# Generate fresh answers with the released checkpoint server.
python -m chartadept.infer --server http://127.0.0.1:8237 \
  --questions ../chartadept-runs/examples/questions.jsonl \
  --output ../chartadept-runs/fresh-examples
```

## Training and evaluation

- [Training](docs/training.md): two SFT stages and GSPO on your own prepared data, with local and Ray runtimes.
- [Evaluation](docs/evaluation.md): question schema, complete-response Compass scoring, benchmark settings, and paired figure bootstrap.
- [Reported results](results/reported-main-table.json): manuscript Table 2 values, identified separately from newly generated evaluation outputs.

The training scripts and paper configurations are provided for adaptation. The private training corpus is required to repeat the original training run; the public ChartQA demonstration exercises the training interface on a different, small dataset.

## Checks

```bash
python -m unittest discover -s tests -v
python -m chartadept.train_sft initial --dry-run
python -m chartadept.train_rl gspo --dry-run
```

These are CPU checks of tools, the interaction loop, grading preparation, statistics, and configuration. They do not train the full model or reproduce benchmark accuracies.

## License and citation

Original code is licensed under [Apache 2.0](LICENSE). See [NOTICE](NOTICE) for upstream projects and separately licensed assets. Use [CITATION.bib](CITATION.bib) to cite the manuscript.
