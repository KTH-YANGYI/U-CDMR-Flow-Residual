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
    parser.add_argument("--mask-source", choices=["bank", "descriptor_flow"], default="descriptor_flow")
    parser.add_argument("--mask-flow-checkpoint", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-type", choices=["residual_flow_dit", "dit", "residual_flow_unet", "unet"], default=None)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--style-dim", type=int, default=None)
    parser.add_argument("--time-dim", type=int, default=None)
    parser.add_argument("--dit-patch-size", type=int, default=None)
    parser.add_argument("--dit-hidden-size", type=int, default=None)
    parser.add_argument("--dit-depth", type=int, default=None)
    parser.add_argument("--dit-num-heads", type=int, default=None)
    parser.add_argument("--dit-mlp-ratio", type=float, default=None)
    parser.add_argument("--max-velocity", type=float, default=None)
    parser.add_argument("--flow-steps", type=int, default=32)
    parser.add_argument("--flow-sampler", choices=["euler", "heun"], default="heun")
    parser.add_argument("--flow-sigma", type=float, default=0.25)
    parser.add_argument("--flow-max-delta", type=float, default=0.25)
    parser.add_argument("--seed-residual", type=int, default=None)
    parser.add_argument("--seed-mask", type=int, default=None)
    parser.add_argument("--mask-match-top-k", type=int, default=5)
    parser.add_argument("--mask-match-max-score", type=float, default=None)
    parser.add_argument("--allow-clamped-masks", action="store_true")
    parser.add_argument("--color-speckle-threshold", type=float, default=0.05)
    parser.add_argument("--support-threshold", type=float, default=8.0)
    parser.add_argument("--teacher-checkpoint", type=Path, default=None)
    parser.add_argument("--teacher-source-root", type=Path, default=None)
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
