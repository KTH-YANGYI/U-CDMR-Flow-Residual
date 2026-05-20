from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from ucdmr_flow_residual_plus.cli.common import add_common_args
from ucdmr_flow_residual_plus.generation import dry_run_filter_summary, filter_synthetic


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Filter plus synthetic image-mask pairs.")
    add_common_args(parser)
    parser.add_argument("--synthetic-manifest", type=Path, default=None)
    parser.add_argument("--filtered-output", type=Path, default=None)
    parser.add_argument("--max-residual-leak", type=float, default=0.1)
    parser.add_argument("--max-outside-change", type=float, default=0.005)
    parser.add_argument("--min-mask-area", type=float, default=1e-6)
    parser.add_argument("--max-mask-area", type=float, default=0.08)
    parser.add_argument("--min-mask-residual-iou", type=float, default=0.01)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.dry_run:
        print(dry_run_filter_summary(args))
        return
    filter_synthetic(args)


if __name__ == "__main__":
    main()
