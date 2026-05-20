from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from ucdmr_flow_residual_plus.cli.common import add_common_args
from ucdmr_flow_residual_plus.generation import dry_run_generate_summary, generate


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Generate plus synthetic image-mask pairs.")
    add_common_args(parser)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--masks-manifest", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--synthetic-output", type=Path, default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--mask-source", choices=["bank", "descriptor_flow"], default="bank")
    parser.add_argument("--residual-source", choices=["flow", "renderer"], default="flow")
    parser.add_argument("--device", default=None)
    parser.add_argument("--tile-size", type=int, default=768)
    parser.add_argument("--tile-overlap", type=int, default=96)
    parser.add_argument("--encoder", default=None)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--style-dim", type=int, default=None)
    parser.add_argument("--max-delta", type=float, default=None)
    parser.add_argument("--time-dim", type=int, default=None)
    parser.add_argument("--max-velocity", type=float, default=None)
    parser.add_argument("--flow-steps", type=int, default=32)
    parser.add_argument("--flow-sampler", choices=["euler", "heun"], default="euler")
    parser.add_argument("--flow-sigma", type=float, default=1.0)
    parser.add_argument("--seed-residual", type=int, default=None)
    parser.add_argument("--seed-mask", type=int, default=None)
    parser.add_argument("--support-threshold", type=float, default=8.0)
    parser.add_argument("--teacher-checkpoint", type=Path, default=None)
    parser.add_argument("--teacher-stage-name", default="teacher_plus")
    parser.add_argument("--teacher-threshold", type=float, default=0.5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.dry_run:
        print(dry_run_generate_summary(args))
        return
    generate(args)


if __name__ == "__main__":
    main()
