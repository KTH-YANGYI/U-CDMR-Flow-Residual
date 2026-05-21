#!/usr/bin/env bash
#SBATCH --job-name=ucdmr-infer-e18-a40
#SBATCH --partition=alvis
#SBATCH --account=naiss2026-3-280
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=A40:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/cephyr/users/yayi/Alvis/ucdmr_slurm_logs/ucdmr-infer-e18-a40-%j.out
#SBATCH --error=/cephyr/users/yayi/Alvis/ucdmr_slurm_logs/ucdmr-infer-e18-a40-%j.err

set -euo pipefail
ROOT="/mimer/NOBACKUP/groups/smart-rail/Yi Yang/CV_contact_wire/ucdmr_flow_residual_plus"
CODE_DIR="$ROOT/code"
DATA_ROOT="$ROOT/data/dataset0505_crop640_roi_dphone"
OUT_ROOT="$ROOT/runs/u_cdmr_flow_residual_plus"
TEACHER_SOURCE_ROOT="/mimer/NOBACKUP/groups/smart-rail/Yi Yang/CV_contact_wire/UNET_two_stage"
TEACHER_CHECKPOINT="$TEACHER_SOURCE_ROOT/outputs/experiments/811_fixed_split/phase2_architecture_full_single_seed/811_m4_skip_d4d3_s20260515/stage2/best_stage2.pt"

cd "$CODE_DIR"
module purge
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
module load SciPy-bundle/2024.05-gfbf-2024a Pillow/10.4.0-GCCcore-13.3.0 PyYAML/6.0.2-GCCcore-13.3.0
export PYTHONPATH="$CODE_DIR/src:${PYTHONPATH:-}"
export TORCH_HOME="$ROOT/pretrained"
export OMP_NUM_THREADS=8

python -m ucdmr_flow_residual_plus.cli.generate_synthetic \
  --dataset-root "$DATA_ROOT" \
  --output-root "$OUT_ROOT" \
  --split-manifest "$OUT_ROOT/data/manifest_splits.csv" \
  --masks-manifest "$OUT_ROOT/masks/masks_manifest.csv" \
  --checkpoint "$OUT_ROOT/residual_flow_plus/checkpoints/epoch_0018.pt" \
  --synthetic-output "$OUT_ROOT/synthetic/inspect_epoch0018_bank" \
  --split train \
  --mask-source bank \
  --flow-steps 32 \
  --flow-sampler heun \
  --flow-sigma 0.35 \
  --flow-max-delta 0.5 \
  --seed-residual 100000 \
  --seed-mask 0 \
  --teacher-checkpoint "$TEACHER_CHECKPOINT" \
  --teacher-source-root "$TEACHER_SOURCE_ROOT" \
  --max-samples 12
