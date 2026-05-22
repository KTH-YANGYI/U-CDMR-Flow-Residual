from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from ucdmr_flow_residual_plus.cli.common import add_common_args
from ucdmr_flow_residual_plus.training.train_residual_flow import dry_run_summary, train


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Train mask-conditioned residual Flow Matching / Rectified Flow.")
    add_common_args(parser)
    parser.add_argument("--pseudo-manifest", type=Path, default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--train-output", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--samples-per-epoch", type=int, default=8192)
    parser.add_argument("--model-type", choices=["residual_flow_dit", "dit", "residual_flow_unet", "unet"], default="residual_flow_dit")
    parser.add_argument("--base-channels", type=int, default=48)
    parser.add_argument("--style-dim", type=int, default=16)
    parser.add_argument("--style-dropout", type=float, default=0.5)
    parser.add_argument("--time-dim", type=int, default=128)
    parser.add_argument("--dit-patch-size", type=int, default=32)
    parser.add_argument("--dit-hidden-size", type=int, default=384)
    parser.add_argument("--dit-depth", type=int, default=8)
    parser.add_argument("--dit-num-heads", type=int, default=6)
    parser.add_argument("--dit-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--max-velocity", type=float, default=1.0)
    parser.add_argument("--flow-sigma", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-flow", type=float, default=1.0)
    parser.add_argument("--lambda-outside", type=float, default=0.25)
    parser.add_argument("--lambda-leak", type=float, default=1.0)
    parser.add_argument("--lambda-tv", type=float, default=0.02)
    parser.add_argument("--lambda-chroma", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="bf16")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.dry_run:
        print(dry_run_summary(args))
        return
    train(args)


if __name__ == "__main__":
    main()
