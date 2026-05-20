#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mimer/NOBACKUP/groups/smart-rail/Yi Yang/CV_contact_wire/ucdmr_flow_residual_plus}"
CODE_DIR="${CODE_DIR:-$ROOT/code}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/dataset0505_crop640_roi_dphone}"
OUT_ROOT="${OUT_ROOT:-$ROOT/runs/u_cdmr_flow_residual_plus}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.12.3-GCCcore-13.3.0}"
EXTRA_MODULES="${EXTRA_MODULES:-SciPy-bundle/2024.05-gfbf-2024a Pillow/10.4.0-GCCcore-13.3.0 PyYAML/6.0.2-GCCcore-13.3.0}"

cd "$CODE_DIR"
module purge
module load "$PYTHON_MODULE"
module load $EXTRA_MODULES
export PYTHONPATH="$CODE_DIR/src:${PYTHONPATH:-}"

python -m ucdmr_flow_residual_plus.cli.prepare_manifest \
  --dataset-root "$DATA_ROOT" \
  --output-root "$OUT_ROOT"

python -m ucdmr_flow_residual_plus.cli.prepare_splits \
  --output-root "$OUT_ROOT"

python -m ucdmr_flow_residual_plus.cli.prepare_masks \
  --dataset-root "$DATA_ROOT" \
  --output-root "$OUT_ROOT" \
  --split train

python -m ucdmr_flow_residual_plus.cli.prepare_pseudo_normal \
  --dataset-root "$DATA_ROOT" \
  --output-root "$OUT_ROOT" \
  --split train \
  --method "${INPAINT_METHOD:-opencv_telea}"

