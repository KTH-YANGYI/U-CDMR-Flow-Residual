#!/usr/bin/env bash
#SBATCH --job-name=ucdmr-infer-e18-bal
#SBATCH --partition=alvis
#SBATCH --account=naiss2026-3-280
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=A100:1
#SBATCH --constraint=4xA100
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/cephyr/users/yayi/Alvis/ucdmr_slurm_logs/ucdmr-infer-e18-bal-%j.out
#SBATCH --error=/cephyr/users/yayi/Alvis/ucdmr_slurm_logs/ucdmr-infer-e18-bal-%j.err

set -euo pipefail
ROOT=/mimer/NOBACKUP/groups/smart-rail/Yi Yang/CV_contact_wire/ucdmr_flow_residual_plus
CODE_DIR=/code
DATA_ROOT=/data/dataset0505_crop640_roi_dphone
OUT_ROOT=/runs/u_cdmr_flow_residual_plus
SYN_OUT=/synthetic/inspect_epoch0018_bank_balanced_a100_
SELECTED_MANIFEST=/selected_balanced_manifest.csv

mkdir -p 
cd 
module purge
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
module load SciPy-bundle/2024.05-gfbf-2024a Pillow/10.4.0-GCCcore-13.3.0 PyYAML/6.0.2-GCCcore-13.3.0
export PYTHONPATH=/src:
export TORCH_HOME=/pretrained
export OMP_NUM_THREADS=8

python - /data/manifest_splits.csv  <<'PY'
import csv
import sys
from collections import defaultdict
src, dst = sys.argv[1], sys.argv[2]
per_domain = 12
buckets = defaultdict(list)
with open(src, newline='') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        if row.get('label') != 'normal':
            continue
        if row.get('split') != 'train':
            continue
        if row.get('label') == 'broken' or 'broken' in row.get('dataset_relative_path', '').lower():
            continue
        domain = row.get('domain') or row.get('dataset_group') or ''
        if domain in {'camera', 'phone', 'dphone'}:
            buckets[domain].append(row)
selected = []
for domain in ['camera', 'phone', 'dphone']:
    selected.extend(buckets[domain][:per_domain])
with open(dst, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(selected)
print({'selected_total': len(selected), 'per_domain_available': {k: len(v) for k, v in buckets.items()}, 'selected_manifest': dst})
PY

python -m ucdmr_flow_residual_plus.cli.generate_synthetic   --dataset-root    --output-root    --split-manifest    --masks-manifest /masks/masks_manifest.csv   --checkpoint /residual_flow_plus/checkpoints/epoch_0018.pt   --synthetic-output    --split train   --mask-source bank   --flow-steps 32   --flow-sampler heun   --flow-sigma 0.35   --flow-max-delta 0.5   --seed-residual 200000   --seed-mask 17   --max-samples 36

echo SYN_OUT=
