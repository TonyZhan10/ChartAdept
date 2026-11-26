#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo 'Usage: bash code/setup_env.sh local | ray NODES GPUS_PER_NODE'
}
case "${1:-}" in
    local)
        [[ $# == 1 ]] || { usage >&2; exit 2; }
        nodes=1
        gpus=8
        ;;
    ray)
        [[ $# == 3 && $2 =~ ^[1-9][0-9]*$ && $3 =~ ^[1-9][0-9]*$ ]] || { usage >&2; exit 2; }
        nodes=$2
        gpus=$3
        ((nodes >= 2)) || { echo 'Ray training needs at least two nodes.' >&2; exit 2; }
        [[ "${RAY_ADDRESS:-auto}" != local ]] || { echo 'Set RAY_ADDRESS to an existing Ray cluster.' >&2; exit 2; }
        ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

mode=$1
code_dir=$(cd -- "$(dirname -- "$0")" && pwd -P)
package_dir=$(dirname -- "$code_dir")
work_dir="${CHARTADEPT_WORK_DIR:-../chartadept-work}"
[[ "$work_dir" == /* ]] || work_dir="$package_dir/$work_dir"
mkdir -p "$work_dir"
work_dir=$(cd -- "$work_dir" && pwd -P)
env_dir="$work_dir/venv-$mode"
python_bin="$env_dir/bin/python"
requirements=("$code_dir/requirements.txt")
if [[ "$mode" == ray ]]; then
    requirements+=("$code_dir/requirements-ray.txt")
fi
export PYTHONDONTWRITEBYTECODE=1
export RAY_USAGE_STATS_ENABLED=0

if [[ ! -x "$python_bin" ]]; then
    python3.12 -m venv "$env_dir"
fi
export PATH="$env_dir/bin:$PATH"
if ! "$python_bin" -B - "${requirements[@]}" <<'PY' 2>/dev/null
import sys
from importlib.metadata import version
from pathlib import Path
from packaging.requirements import Requirement

requirements = ['flash-attn==2.8.3']
for filename in sys.argv[1:]:
    requirements.extend(line.partition('#')[0].strip() for line in Path(filename).read_text().splitlines())
for requirement in filter(None, requirements):
    req = Requirement(requirement)
    if not req.specifier.contains(version(req.name)):
        sys.exit(1)
PY
then
    if ! command -v uv >/dev/null; then
        "$python_bin" -m pip install 'uv==0.10.4'
    fi
    install_args=()
    for file in "${requirements[@]}"; do install_args+=(-r "$file"); done
    uv pip install --python "$python_bin" "${install_args[@]}"
    MAX_JOBS="${MAX_JOBS:-4}" "$python_bin" -m pip install --no-build-isolation 'flash-attn==2.8.3'
fi

cd "$code_dir"
"$python_bin" -B - "$mode" "$nodes" "$gpus" "$work_dir" <<'PY'
import os
from pathlib import Path
import sys
import torch
from flash_attn import flash_attn_varlen_func
from llamafactory.train.tuner import run_exp
from chartadept.io import write_json

mode, nodes, gpus, work_dir = sys.argv[1:]
settings = {'mode': mode, 'nodes': int(nodes), 'gpus_per_node': int(gpus)}
if torch.version.cuda != '12.8':
    raise SystemExit(f'Expected PyTorch with CUDA 12.8, found {torch.__version__}.')
if mode == 'ray':
    from verl.trainer.main_ppo import run_ppo
    from vllm import _C
    from chartadept.ray_runtime import check_cluster
    settings['address'] = os.environ.get('RAY_ADDRESS', 'auto')
    check_cluster(settings, Path(work_dir))
write_json(Path(work_dir) / 'settings.json', settings)
print(f'Ready for {mode} training on {nodes} node(s), {gpus} GPUs per node, with {sys.executable}.')
PY
