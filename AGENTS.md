# U-CDMR-Flow-Residual Agent Notes

## Core Method

This project is for domain-conditioned device/screen crack image-mask pair synthesis.

Keep the main generation formula fixed:

```text
I_syn = I_normal + gate(M_syn) * Delta_flow
Y_syn = M_syn
```

- `I_normal` comes from a same-domain real normal image.
- `M_syn` comes from a same-domain mask source, preferably descriptor mask flow.
- `Delta_flow` comes from residual rectified flow ODE sampling.
- `Y_syn` is exactly the generated/input mask, not a mask predicted back from the synthetic RGB image.
- Mask-outside background should remain essentially unchanged.

Do not introduce FLUX, Stable Diffusion, VAE/LDM full-image diffusion, full RGB generation, cross-domain mixing, or `broken` samples.

## Domain Rules

The base residual-flow domains are:

```text
camera
phone
dphone
```

For mask descriptor flow and final synthesis matching, split `dphone` into two effective domains because the camera/viewpoint changes the mask placement distribution:

```text
dphone id < 270   -> dphone_lt270
dphone id >= 270  -> dphone_ge270
```

ID `270` is assigned to `dphone_ge270` so no sample is left unassigned.

This split is required for:

- mask descriptor flow training
- sampled mask manifest domain labels
- normal-to-mask matching during synthesis
- per-domain generation/filter summaries

Current residual flow checkpoints are still three-domain checkpoints. When using an existing residual checkpoint, both `dphone_lt270` and `dphone_ge270` must map back to the base residual domain `dphone`. Only retrain residual flow with four domain embeddings if visual inspection shows the two dphone views need distinct residual texture/intensity modeling.

## Training Before Downstream

Before downstream segmentation experiments, the synthesis side has these stages:

1. Prepare real masks and pseudo-normal images from real crack samples.
2. Train or reuse the teacher segmentation model. The preferred teacher is the existing `m4, d3d4 skip gate` checkpoint under the Alvis `CV_contact_wire` area.
3. Train residual flow on real crack plus pseudo-normal pairs.
4. Train descriptor mask flow with the effective domains above.
5. Sample same-domain masks from descriptor mask flow.
6. Generate synthetic image-mask pairs with residual flow:

```text
same-domain normal + same-domain generated mask -> residual flow -> synthetic image
```

7. Filter synthetic samples by residual leakage, outside background change, mask area, mask-residual alignment, and optional teacher scores.
8. Only then train downstream segmentation comparisons.

Mask bank is a diagnostic/baseline mask source because it reuses real masks. The main route should be descriptor mask flow plus residual flow.

## Alvis Safety

- Do not start downstream segmentation training unless explicitly requested.
- Do not allocate extra GPUs beyond the current request.
- Keep new training/inference outputs in a fresh run directory; do not mix them into previous runs.
- Prefer finite residual checkpoints, especially `latest_finite.pt`.
- Never generate from a non-finite residual checkpoint.
