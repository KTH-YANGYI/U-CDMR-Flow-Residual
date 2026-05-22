from __future__ import annotations

from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

from ucdmr_flow_residual_plus.cli.common import add_common_args
from ucdmr_flow_residual_plus.mask_descriptor_flow import dry_run_sample_summary, sample


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Sample plus masks from descriptor-level Mask Flow.")
    add_common_args(parser)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--template-manifest", type=Path, default=None)
    parser.add_argument("--sample-output", type=Path, default=None)
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--samples-per-domain", type=int, default=None)
    parser.add_argument("--domain", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--min-area-ratio", type=float, default=1e-6)
    parser.add_argument("--max-area-ratio", type=float, default=0.08)
    parser.add_argument("--inpaint-radius", type=int, default=None)
    parser.add_argument("--band-radius", type=int, default=None)
    parser.add_argument("--gate-radius", type=int, default=None)
    parser.add_argument("--gate-blur", type=float, default=None)
    parser.add_argument("--max-render-center-error", type=float, default=0.03)
    parser.add_argument("--max-render-angle-error", type=float, default=15.0)
    parser.add_argument("--reject-clamped", action=BooleanOptionalAction, default=True)
    parser.add_argument("--keep-rejected", action="store_true", help="write rejected masks to sampled_masks_manifest_all.csv for debugging")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker-start-method", choices=["spawn", "fork", "forkserver"], default="spawn")
    parser.add_argument("--progress-every", type=int, default=25)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.dry_run:
        print(dry_run_sample_summary(args))
        return
    sample(args)


if __name__ == "__main__":
    main()
