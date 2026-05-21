# Alvis U-CDMR Flow Training and Inference Logs

This directory archives the Slurm logs and job scripts from the first U-CDMR-Flow-Residual residual-flow run and the follow-up inference checks on Alvis.

## Training Jobs

| Job ID | Files | Notes |
| --- | --- | --- |
| `6656197` | `ucdmr-plus-full-6656197.out`, `ucdmr-plus-full-6656197.err` | First full pipeline attempt. It was cancelled after spending GPU allocation time in single-process CPU preprocessing. |
| `6656213` | `ucdmr-plus-full-6656213.out`, `ucdmr-plus-full-6656213.err` | Main residual-flow training run after 64-worker preprocessing. Training became non-finite at `epoch=19`, first logged at `global_step=19560`. |

Important checkpoint interpretation from `6656213`:

- Last known finite checkpoint: `epoch_0018.pt`
- First saved non-finite checkpoint: `epoch_0019.pt`
- `latest.pt` was overwritten by a non-finite checkpoint and should not be used

## Inference Jobs

| Job ID | Files | Notes |
| --- | --- | --- |
| `6656451` | `ucdmr-infer-e18-6656451.out`, `ucdmr-infer-e18-6656451.err`, `ucdmr-infer-epoch0018-bank.slurm` | Early inference submission attempt. |
| `6656453` | `ucdmr-infer-e18-a40-6656453.out`, `ucdmr-infer-e18-a40-6656453.err`, `ucdmr-infer-epoch0018-bank-a40.slurm` | A40 submission attempt. It should not be repeated for this workflow unless explicitly requested. |
| `6656454` | `ucdmr-infer-e18-a100-6656454.out`, `ucdmr-infer-e18-a100-6656454.err`, `ucdmr-infer-epoch0018-bank-a100.slurm` | Successful one-A100 inference with `epoch_0018.pt`, 12 bank-mask samples. |
| `6656461` | `ucdmr-infer-e18-bal-6656461.out`, `ucdmr-infer-e18-bal-6656461.err`, `ucdmr-infer-e18-balanced-a100.slurm` | Failed one-A100 balanced inference attempt due to shell expansion of the remote path containing `Yi Yang`. |
| `6656462` | `ucdmr-infer-e18-bal-6656462.out`, `ucdmr-infer-e18-bal-6656462.err`, `ucdmr-infer-e18-balanced-a100-v2.slurm` | Successful one-A100 balanced inference with `epoch_0018.pt`, 36 samples: 12 camera, 12 phone, 12 dphone. |

## Useful Grep Commands

```bash
grep -nEi "nan|inf|error|traceback|runtime|overflow" \
  logs/alvis/2026-05-20_ucdmr_flow_training/ucdmr-plus-full-6656213.out \
  logs/alvis/2026-05-20_ucdmr_flow_training/ucdmr-plus-full-6656213.err
```

```bash
grep -n "global_step" logs/alvis/2026-05-20_ucdmr_flow_training/ucdmr-plus-full-6656213.out
```

## Related Local Artifacts

The generated image inspection artifacts are intentionally not tracked in git. Locally they were saved under:

- `artifacts/inspect_epoch0018_bank_a100_6656454/`
- `artifacts/inspect_epoch0018_bank_balanced_a100_6656462_result/`
