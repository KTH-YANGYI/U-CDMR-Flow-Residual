from __future__ import annotations

from argparse import ArgumentParser

from ucdmr_flow_residual_plus.cli.train_teacher import add_segmentation_args
from ucdmr_flow_residual_plus.training.train_segmentation import dry_run_summary, train


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Train plus downstream segmenter with optional synthetic data.")
    add_segmentation_args(parser, include_synthetic_default=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.dry_run:
        print(dry_run_summary(args, stage_name="downstream_plus"))
        return
    train(args, stage_name="downstream_plus")


if __name__ == "__main__":
    main()

