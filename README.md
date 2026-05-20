# U-CDMR-Flow-Residual+

This repository is now centered on the `U-CDMR-Flow-Residual+` pipeline for native-resolution device/screen crack synthesis.

The active method is:

```text
same-domain normal image
+ domain-conditioned crack mask
+ mask-gated crack residual
= synthetic crack image + synthetic binary mask
```

Core formula:

```text
I_syn = I_normal + gate(M_syn) * Delta_crack
Y_syn = M_syn
```

The generator does not repaint the full RGB image. It predicts only the local crack residual and blends it through a mask gate.

## Dataset Rules

Default dataset:

```text
/Users/yangyi/Desktop/masterthesis/dataset0505_crop640_roi_dphone
```

Only these labels are used:

```text
crack
normal
```

`broken` is ignored for every active stage.

Native resolution is preserved:

```text
camera: 640x640
phone:  640x640
dphone: 1408x2560
```

dphone is not globally cropped or resized. Training uses full native images. Mixed-size batches are padded only inside the batch, and padding is ignored by the loss.

## Active Pipeline

```text
manifest_merged.csv
  -> filter broken
  -> domain/video split
  -> LabelMe to masks
  -> M_raw / M_inpaint / M_band / M_gate / skeleton / SDF / thickness
  -> pseudo-normal from real crack images
  -> residual Flow Matching / Rectified Flow
  -> optional descriptor Mask Flow
  -> synthetic image-mask pairs
  -> synthetic filter
  -> existing teacher checkpoint / downstream segmentation
  -> real held-out evaluation
```

The active package is:

```text
src/ucdmr_flow_residual_plus/
```

Before using console commands locally, either install the repo or export `PYTHONPATH`:

```bash
python -m pip install -e .
# or
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
```

Default output:

```text
artifacts/dataset0505_crop640_roi_dphone/methods/u_cdmr_flow_residual_plus/
```

## Local Commands

Data preparation:

```bash
ucdmr_plus_prepare_manifest
ucdmr_plus_prepare_splits
ucdmr_plus_prepare_masks
ucdmr_plus_domain_stats
ucdmr_plus_prepare_pseudo_normal
```

Training and generation:

```bash
# optional only if no existing segmentation teacher is available
ucdmr_plus_train_teacher
ucdmr_plus_train_residual_flow
ucdmr_plus_train_mask_descriptor_flow
ucdmr_plus_sample_masks
ucdmr_plus_generate_synthetic
ucdmr_plus_filter_synthetic
ucdmr_plus_train_downstream
ucdmr_plus_eval_downstream
```

Dry-run smoke checks:

```bash
python -m compileall src
ucdmr_plus_prepare_manifest --dry-run
ucdmr_plus_prepare_splits --dry-run
ucdmr_plus_prepare_masks --dry-run --split train --max-samples 1
ucdmr_plus_prepare_pseudo_normal --dry-run --split train --max-samples 1
ucdmr_plus_train_teacher --dry-run
ucdmr_plus_train_residual_flow --dry-run
ucdmr_plus_train_mask_descriptor_flow --dry-run
ucdmr_plus_sample_masks --dry-run
ucdmr_plus_generate_synthetic --dry-run
ucdmr_plus_filter_synthetic --dry-run
ucdmr_plus_train_downstream --dry-run
ucdmr_plus_eval_downstream --dry-run
```

## Model Route

Residual Flow Matching main generator:

```text
x1 = gate(M_real) * (I_crack - I_pseudo_normal)
x0 = gate(M_real) * gaussian_noise
x_t = (1 - t) * x0 + t * x1
FlowModel(x_t, t, I_pseudo_normal, mask representations, domain, style)
  -> velocity
```

Generation samples `Delta_flow` by ODE integration and blends it as:

```text
I_syn = I_normal + gate(M_syn) * Delta_flow
Y_syn = M_syn
```

Teacher/downstream segmenter:

```text
image
  -> pretrained visual encoder
  -> segmentation decoder
  -> crack probability mask
```

Mask generator V1:

```text
descriptor-level mask flow
+ same-domain real mask template bank
+ resize / rotate / place
```

No FLUX / Stable Diffusion / full-image DiT is used in the active route.

## Alvis

Prepare plus artifacts:

```bash
bash scripts/alvis/prepare_ucdmr_plus_data.sh
```

Train residual flow:

```bash
sbatch scripts/alvis/train_ucdmr_plus_residual_flow_2node_8gpu.slurm
```

Teacher segmentation does not need to be retrained if an existing checkpoint is available. The Alvis generation script can load legacy `UNET_two_stage` checkpoints through `TEACHER_CHECKPOINT` and `TEACHER_SOURCE_ROOT`; by default it uses the existing native-resolution stage-2 `m4_skip_d4d3` checkpoint under `UNET_two_stage/outputs/experiments/811_fixed_split/phase2_architecture_full_single_seed/811_m4_skip_d4d3_s20260515/stage2/best_stage2.pt`.

Optional descriptor Mask Flow:

```bash
bash scripts/alvis/train_ucdmr_plus_mask_descriptor_flow.sh
bash scripts/alvis/sample_ucdmr_plus_masks.sh
```

Generate/filter synthetic pairs:

```bash
bash scripts/alvis/generate_ucdmr_plus_synthetic.sh
bash scripts/alvis/filter_ucdmr_plus_synthetic.sh
```

Descriptor-flow masks are the default generation route. Use mask-bank masks for a conservative baseline:

```bash
MASK_SOURCE=bank bash scripts/alvis/generate_ucdmr_plus_synthetic.sh
```

Downstream comparison:

```bash
# real-only
sbatch scripts/alvis/train_ucdmr_plus_downstream_2node_8gpu.slurm

# real + synthetic
USE_SYNTHETIC=1 sbatch scripts/alvis/train_ucdmr_plus_downstream_2node_8gpu.slurm

# evaluation on real held-out data
bash scripts/alvis/eval_ucdmr_plus_downstream.sh
```

The non-`sbatch` Alvis scripts refuse to run outside a Slurm allocation unless `ALLOW_LOGIN_NODE=1` is set for a tiny manual check.

The Slurm scripts request two nodes and four A100 GPUs per node by default. Add the correct `#SBATCH --account=...` line before submitting if your allocation requires it.
