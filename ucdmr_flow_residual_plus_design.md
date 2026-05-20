# U-CDMR-Flow-Residual+ Design

## Goal

Generate usable crack segmentation image-mask pairs while keeping the image generation problem local and mask-aligned:

```text
same-domain normal image
+ domain-conditioned crack mask
+ mask-gated crack residual sampled by residual flow
= synthetic crack image + synthetic binary mask
```

The method does not generate full RGB images, does not infer masks from generated images, does not use `broken`, and does not use FLUX / Stable Diffusion / LDM / full-image DiT as the main generator.

## Core Formula

```text
I_syn = I_normal + gate(M_syn) * Delta_flow
Y_syn = M_syn
```

`Y_syn` is always the input/generated crack mask. The synthetic label is not predicted from the generated image.

## Data Flow

```text
manifest_merged.csv
  -> filter label in {crack, normal}
  -> video-level train/val/test split
  -> real crack annotation to mask bank
  -> pseudo-normal construction from train crack images
  -> residual Flow Matching training
  -> same-domain synthetic generation
  -> synthetic filtering
  -> downstream segmentation comparison
```

`broken` is never used.

Native output canvas is preserved:

```text
camera: 640x640
phone:  640x640
dphone: 1408x2560
```

Training uses full native-resolution images. Mixed-size batches may be padded inside the batch, but padding is masked out of the loss.

## Mask Branch

The mask branch is still mask-first.

Supported mask sources:

```text
mask bank
descriptor-level mask flow
```

The descriptor mask flow only generates low-dimensional mask descriptors, then renders them through real mask templates. It does not generate RGB and it does not replace the downstream label.

Same-domain generation is mandatory:

```text
camera normal + camera mask -> camera synthetic
phone normal  + phone mask  -> phone synthetic
dphone normal + dphone mask -> dphone synthetic
```

## Residual Flow Branch

The main image residual generator is now residual Flow Matching / Rectified Flow.

Training pairs come from real crack images:

```text
I_crack + M_real
  -> erase/inpaint crack
  -> I_ctx
```

Flow target:

```text
x1 = gate(M_real) * (I_crack - I_ctx)
x0 = gate(M_real) * gaussian_noise
x_t = (1 - t) * x0 + t * x1
v_target = x1 - x0
```

Model:

```text
FlowModel(
  x_t,
  t,
  I_ctx,
  M_raw / M_inpaint / M_band / M_gate / skeleton / SDF / thickness,
  domain,
  style
) -> v_pred
```

Sampling:

```text
x_0 = gate(M_syn) * gaussian_noise
ODE integrate v_pred from t=0 to t=1
Delta_flow = x_1_sampled
I_syn = I_normal + gate(M_syn) * Delta_flow
```

Mask outside the gate should remain unchanged by construction and by filtering.

## Synthetic Manifest Requirements

Each generated sample records:

```text
residual_source = flow
mask_source
residual_flow_checkpoint
flow_steps
flow_sampler
flow_sigma
seed_residual
seed_mask
native_width
native_height
image_path
mask_path
residual_path
```

## Validation

Required checks:

```text
python -m compileall src scripts
ucdmr_plus_train_residual_flow --dry-run
ucdmr_plus_generate_synthetic --dry-run
```

Main downstream comparison:

```text
real only
real + mask flow + residual flow synthetic
```

Report by domain and crack size bucket:

```text
Dice
IoU
Recall
Boundary F1
normal false positive rate
```

The main result should come from residual flow, not only from descriptor-level mask flow.
