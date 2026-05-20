from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from ucdmr_flow_residual_plus.cli.common import add_common_args
from ucdmr_flow_residual_plus.training.train_segmentation import dry_run_summary, train


def add_segmentation_args(parser: ArgumentParser, *, include_synthetic_default: bool) -> None:
    add_common_args(parser)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--synthetic-manifest", type=Path, default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--train-output", type=Path, default=None)
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", action="store_false", dest="pretrained")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tile-size", type=int, default=768)
    parser.add_argument("--samples-per-epoch", type=int, default=4096)
    parser.add_argument("--base-channels", type=int, default=48)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--focal-weight", type=float, default=0.5)
    parser.add_argument("--pos-weight", type=float, default=8.0)
    parser.add_argument("--synthetic-weight", type=int, default=1)
    parser.add_argument("--include-synthetic", action="store_true", dest="include_synthetic", default=include_synthetic_default)
    parser.add_argument("--no-synthetic", action="store_false", dest="include_synthetic")
    parser.add_argument("--require-synthetic", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Train plus teacher segmenter with a pretrained encoder.")
    add_segmentation_args(parser, include_synthetic_default=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.dry_run:
        print(dry_run_summary(args, stage_name="teacher_plus"))
        return
    train(args, stage_name="teacher_plus")


if __name__ == "__main__":
    main()

