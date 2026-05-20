# U-CDMR-Flow-Residual+

This repository is now centered on the `U-CDMR-Flow-Residual+` pipeline for native-resolution contact-wire crack synthesis.

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

dphone is not globally cropped or resized. Training may use tiles, but generated images, masks, predictions, and evaluation outputs stay on the native canvas.

## Active Pipeline

```text
manifest_merged.csv
  -> filter broken
  -> domain/video split
  -> LabelMe to masks
  -> M_raw / M_inpaint / M_band / M_gate / skeleton / SDF / thickness
  -> pseudo-normal from real crack images
  -> pretrained-encoder residual renderer
  -> optional descriptor Mask Flow
  -> synthetic image-mask pairs
  -> synthetic filter
  -> teacher/downstream segmentation
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
ucdmr_plus_train_teacher
ucdmr_plus_train_residual_renderer
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
ucdmr_plus_train_residual_renderer --dry-run
ucdmr_plus_train_mask_descriptor_flow --dry-run
ucdmr_plus_sample_masks --dry-run
ucdmr_plus_generate_synthetic --dry-run
ucdmr_plus_filter_synthetic --dry-run
ucdmr_plus_train_downstream --dry-run
ucdmr_plus_eval_downstream --dry-run
```

## Model Route

Residual renderer:

```text
I_context + mask representations + domain + style noise
  -> pretrained visual encoder
  -> residual decoder
  -> Delta_rgb
```

The default encoder is ImageNet-pretrained `resnet34`. If pretrained weights are not cached on the node and cannot be loaded, training fails fast instead of silently switching to a random encoder. Use `--no-pretrained` only when you intentionally want a scratch baseline.

Teacher/downstream segmenter:

```text
image
  -> pretrained visual encoder
  -> segmentation decoder
  -> crack probability mask
```

Mask generator V1:

```text
real mask template bank
+ descriptor-level flow matching
+ resize / rotate / place
```

No FLUX / Stable Diffusion / full-image DiT is used in the active route.

## Alvis

Prepare plus artifacts:

```bash
bash scripts/alvis/prepare_ucdmr_plus_data.sh
```

Train teacher and residual renderer:

```bash
sbatch scripts/alvis/train_ucdmr_plus_teacher_2node_8gpu.slurm
sbatch scripts/alvis/train_ucdmr_plus_residual_renderer_2node_8gpu.slurm
```

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

Use descriptor-flow masks during generation:

```bash
MASK_SOURCE=descriptor_flow bash scripts/alvis/generate_ucdmr_plus_synthetic.sh
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
