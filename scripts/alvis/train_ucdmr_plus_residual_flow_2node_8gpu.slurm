#!/usr/bin/env bash
#SBATCH --job-name=ucdmr-plus-flow
#SBATCH --partition=alvis
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=A100:4
#SBATCH --constraint=4xA100
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
##SBATCH --account=CHANGE_ME

set -euo pipefail

ROOT="${ROOT:-/mimer/NOBACKUP/groups/smart-rail/Yi Yang/CV_contact_wire/ucdmr_flow_residual_plus}"
CODE_DIR="${CODE_DIR:-$ROOT/code}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/dataset0505_crop640_roi_dphone}"
OUT_ROOT="${OUT_ROOT:-$ROOT/runs/u_cdmr_flow_residual_plus}"
PYTORCH_MODULE="${PYTORCH_MODULE:-PyTorch/2.7.1-foss-2024a-CUDA-12.6.0}"
EXTRA_MODULES="${EXTRA_MODULES:-SciPy-bundle/2024.05-gfbf-2024a Pillow/10.4.0-GCCcore-13.3.0 PyYAML/6.0.2-GCCcore-13.3.0}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29514}"

cd "$CODE_DIR"
module purge
module load "$PYTORCH_MODULE"
module load $EXTRA_MODULES
export PYTHONPATH="$CODE_DIR/src:${PYTHONPATH:-}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/pretrained}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

MASTER_ADDR="${PROEPI_HEAD_NODE:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)}"
NNODES="${SLURM_NNODES:-2}"

srun --ntasks="$NNODES" --ntasks-per-node=1 bash -c '
  torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node="'"$GPUS_PER_NODE"'" \
    --node_rank="$SLURM_PROCID" \
    --rdzv_backend=c10d \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_endpoint="'"$MASTER_ADDR:$MASTER_PORT"'" \
    -m ucdmr_flow_residual_plus.cli.train_residual_flow \
      --dataset-root "'"$DATA_ROOT"'" \
      --output-root "'"$OUT_ROOT"'" \
      --pseudo-manifest "'"$OUT_ROOT"'/pseudo_normal/pseudo_manifest.csv" \
      --split train \
      --epochs "${EPOCHS:-30}" \
      --batch-size "${BATCH_SIZE:-1}" \
      --workers "${WORKERS:-8}" \
      --samples-per-epoch "${SAMPLES_PER_EPOCH:-8192}" \
      --flow-sigma "${FLOW_SIGMA:-0.35}" \
      --amp
'
