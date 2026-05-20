from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from ucdmr_flow_residual_plus.cli.common import add_common_args
from ucdmr_flow_residual_plus.mask_descriptor_flow import dry_run_train_summary, train


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Train plus descriptor-level Mask Flow.")
    add_common_args(parser)
    parser.add_argument("--masks-manifest", type=Path, default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--train-output", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.dry_run:
        print(dry_run_train_summary(args))
        return
    train(args)


if __name__ == "__main__":
    main()

