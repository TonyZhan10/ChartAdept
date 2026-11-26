# Training on user-provided data

The release includes both SFT stages, terminal-reward GSPO, and the Zoom/Python runtime. It excludes the paper's 90,474 training trajectories, source collection, Guide/Solver construction prompts, chart-fact extraction, and demonstration admission/review pipeline.

## Small runnable training example

Use the existing installer from the repository root. It creates a sibling working directory for environments, downloaded assets, and runs.

```bash
bash setup_env.sh local
python3.12 -B -m chartadept.run sft --data /path/to/your-data --epochs 1
python3.12 -B -m chartadept.run gspo --updates 3
python3.12 -B -m chartadept.run infer
```

The local example uses eight GPUs. For an existing two-node Ray cluster with four GPUs per node:

```bash
bash setup_env.sh ray 2 4
python3.12 -B -m chartadept.run sft --data /shared/path/to/your-data
python3.12 -B -m chartadept.run gspo --updates 3
python3.12 -B -m chartadept.run infer
```

Mount the code, data, model, and sibling working directory at the same paths on cluster nodes. Configure `RAY_ADDRESS` if the cluster is not discovered automatically. `CHARTADEPT_WORK_DIR` sets the working directory; `BASE_MODEL` selects a local base checkpoint.

Omitting `--data` downloads a small public ChartQA example: 64 SFT questions, a separate 256-question GSPO pool, and eight questions each for development and inference. This checks the training interface and uses one epoch per SFT stage and three GSPO updates by default. It is not the paper training corpus or schedule.

## Input format

Provide `train.jsonl`, `development.jsonl`, and `test.jsonl`. Optionally provide `gspo.jsonl`; otherwise GSPO uses the training question pool. Each row uses:

```json
{"id":"example-1","image":"images/example.png","question":"Your question","prompt":"Your formatted question","reference_answer":"Your reference","task":"chart-qa"}
```

Images are relative to this input directory. Training rows may include a `messages` list containing your existing assistant demonstrations and actual tool observations. Without `messages`, the example loader formats a direct answer from the supplied reference. This formatting helper does not generate or verify research trajectories.

The generic loader writes LLaMA-Factory SFT files and VERL question parquet files in the run directory. Keep question IDs and chart groups disjoint across train, development, and test.

For nonnumeric GSPO targets, supply a semantic reward endpoint using `--api-config /private/path/reward-api.json`. Adapt `configs/api.example.json`; store real credentials outside the repository. Numeric demonstrations can use scalar numeric-reference rewards. Benchmark Compass grading is configured separately.

## Paper configurations

`configs/sft-initial.yaml` and `configs/sft-continue.yaml` retain the reported two three-epoch stages. Their saved full-scale paths expect the original prepared cache and route schedules. Those private data artifacts are not distributed.

`configs/gspo.yaml` and `configs/gspo-recipe.json` retain the full GSPO configuration, including K=16, sequence-ratio clipping, zero KL coefficient, and terminal rewards. To use your own prepared GSPO parquet files and SFT export:

```bash
python -m chartadept.train_rl gspo \
  --model /path/to/your-sft-export --data /path/to/your-parquet-directory \
  --api-config /private/path/reward-api.json \
  --output /path/to/your-run --dry-run
```

Inspect the resolved configuration, then run without `--dry-run` on the configured GPU topology. The directory must contain `train.parquet` and `development.parquet` with the schema produced by the generic loader.

For raw SFT demonstrations, `chartadept.train_sft --dataset-dir` accepts the loader's LLaMA-Factory directory and uses the documented smaller raw-data training path. `--model`, `--output`, and `--resume` select paths explicitly. This path changes batching and sequence length for the small example; it does not silently claim the full-scale recipe.
