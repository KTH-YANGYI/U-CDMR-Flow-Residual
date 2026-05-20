#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mimer/NOBACKUP/groups/smart-rail/Yi Yang/CV_contact_wire/ucdmr_flow_residual_plus}"
CODE_DIR="${CODE_DIR:-$ROOT/code}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/dataset0505_crop640_roi_dphone}"
OUT_ROOT="${OUT_ROOT:-$ROOT/runs/u_cdmr_flow_residual_plus}"
PYTORCH_MODULE="${PYTORCH_MODULE:-PyTorch/2.7.1-foss-2024a-CUDA-12.6.0}"
EXTRA_MODULES="${EXTRA_MODULES:-SciPy-bundle/2024.05-gfbf-2024a Pillow/10.4.0-GCCcore-13.3.0 PyYAML/6.0.2-GCCcore-13.3.0}"
MASK_SOURCE="${MASK_SOURCE:-bank}"

if [[ -z "${SLURM_JOB_ID:-}" && "${ALLOW_LOGIN_NODE:-0}" != "1" ]]; then
  echo "Refusing to run synthetic generation outside a Slurm allocation. Use salloc/sbatch, or set ALLOW_LOGIN_NODE=1 for a small dry check." >&2
  exit 2
fi

cd "$CODE_DIR"
module purge
module load "$PYTORCH_MODULE"
module load $EXTRA_MODULES
export PYTHONPATH="$CODE_DIR/src:${PYTHONPATH:-}"

MASKS_MANIFEST="$OUT_ROOT/masks/masks_manifest.csv"
if [[ "$MASK_SOURCE" == "descriptor_flow" ]]; then
  MASKS_MANIFEST="$OUT_ROOT/sampled_masks/sampled_masks_manifest.csv"
fi
CHECKPOINT="$OUT_ROOT/residual_flow_plus/checkpoints/latest.pt"

python -m ucdmr_flow_residual_plus.cli.generate_synthetic \
  --dataset-root "$DATA_ROOT" \
  --output-root "$OUT_ROOT" \
  --split-manifest "$OUT_ROOT/data/manifest_splits.csv" \
  --masks-manifest "$MASKS_MANIFEST" \
  --checkpoint "$CHECKPOINT" \
  --split train \
  --mask-source "$MASK_SOURCE" \
  --flow-steps "${FLOW_STEPS:-32}" \
  --flow-sampler "${FLOW_SAMPLER:-euler}" \
  --flow-sigma "${FLOW_SIGMA:-1.0}" \
  --seed-residual "${SEED_RESIDUAL:-100000}" \
  --seed-mask "${SEED_MASK:-0}" \
  --tile-size "${TILE_SIZE:-768}" \
  --tile-overlap "${TILE_OVERLAP:-96}" \
  --max-samples "${MAX_SAMPLES:-100}"
