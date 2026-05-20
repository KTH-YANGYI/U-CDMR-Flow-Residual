from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from ucdmr_flow_residual_plus.cli.common import add_common_args
from ucdmr_flow_residual_plus.evaluation import dry_run_summary, evaluate


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Evaluate plus downstream segmenter on real data.")
    add_common_args(parser)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--eval-output", type=Path, default=None)
    parser.add_argument("--stage-name", default="downstream_plus")
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default=None)
    parser.add_argument("--tile-size", type=int, default=768)
    parser.add_argument("--tile-overlap", type=int, default=128)
    parser.add_argument("--encoder", default=None)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save-predictions", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.dry_run:
        print(dry_run_summary(args))
        return
    evaluate(args)


if __name__ == "__main__":
    main()
