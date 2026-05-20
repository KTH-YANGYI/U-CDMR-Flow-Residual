#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mimer/NOBACKUP/groups/smart-rail/Yi Yang/CV_contact_wire/ucdmr_flow_residual_plus}"
CODE_DIR="${CODE_DIR:-$ROOT/code}"
OUT_ROOT="${OUT_ROOT:-$ROOT/runs/u_cdmr_flow_residual_plus}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.12.3-GCCcore-13.3.0}"
EXTRA_MODULES="${EXTRA_MODULES:-SciPy-bundle/2024.05-gfbf-2024a PyYAML/6.0.2-GCCcore-13.3.0}"

cd "$CODE_DIR"
module purge
module load "$PYTHON_MODULE"
module load $EXTRA_MODULES
export PYTHONPATH="$CODE_DIR/src:${PYTHONPATH:-}"

python -m ucdmr_flow_residual_plus.cli.filter_synthetic \
  --output-root "$OUT_ROOT" \
  --synthetic-manifest "$OUT_ROOT/synthetic/raw/synthetic_manifest.csv"

