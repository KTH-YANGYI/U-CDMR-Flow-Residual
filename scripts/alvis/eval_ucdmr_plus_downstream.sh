#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mimer/NOBACKUP/groups/smart-rail/Yi Yang/CV_contact_wire/ucdmr_flow_residual_plus}"
CODE_DIR="${CODE_DIR:-$ROOT/code}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/dataset0505_crop640_roi_dphone}"
OUT_ROOT="${OUT_ROOT:-$ROOT/runs/u_cdmr_flow_residual_plus}"
PYTORCH_MODULE="${PYTORCH_MODULE:-PyTorch/2.7.1-foss-2024a-CUDA-12.6.0}"
EXTRA_MODULES="${EXTRA_MODULES:-SciPy-bundle/2024.05-gfbf-2024a Pillow/10.4.0-GCCcore-13.3.0 PyYAML/6.0.2-GCCcore-13.3.0}"

if [[ -z "${SLURM_JOB_ID:-}" && "${ALLOW_LOGIN_NODE:-0}" != "1" ]]; then
  echo "Refusing to run downstream evaluation outside a Slurm allocation. Use salloc/sbatch, or set ALLOW_LOGIN_NODE=1 for a small dry check." >&2
  exit 2
fi

cd "$CODE_DIR"
module purge
module load "$PYTORCH_MODULE"
module load $EXTRA_MODULES
export PYTHONPATH="$CODE_DIR/src:${PYTHONPATH:-}"

python -m ucdmr_flow_residual_plus.cli.eval_downstream \
  --dataset-root "$DATA_ROOT" \
  --output-root "$OUT_ROOT" \
  --split-manifest "$OUT_ROOT/data/manifest_splits.csv" \
  --stage-name "${STAGE_NAME:-downstream_plus}" \
  --split "${SPLIT:-test}" \
  --tile-size "${TILE_SIZE:-768}" \
  --tile-overlap "${TILE_OVERLAP:-128}" \
  --save-predictions
