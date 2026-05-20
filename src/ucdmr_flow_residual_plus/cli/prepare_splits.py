from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from ucdmr_flow_residual_plus.io_utils import read_csv_records, write_csv_records, write_json

from ucdmr_flow_residual_plus.cli.common import add_common_args
from ucdmr_flow_residual_plus.config import load_config, resolve_output_root
from ucdmr_flow_residual_plus.splits import assign_video_splits, summarize_splits


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Prepare domain + video-level splits for U-CDMR-Flow-Residual+.")
    add_common_args(parser)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    output_root = resolve_output_root(config, args.output_root)
    manifest_path = args.manifest or output_root / "data" / "manifest_filtered.csv"
    if args.dry_run and not Path(manifest_path).exists():
        print(
            {
                "stage": "plus_prepare_splits",
                "source_manifest": str(manifest_path),
                "output": str(output_root / "data" / "manifest_splits.csv"),
                "dry_run": True,
                "input_exists": False,
            }
        )
        return
    rows = read_csv_records(manifest_path)
    rows = assign_video_splits(
        rows,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    summary = {
        "stage": "plus_prepare_splits",
        "source_manifest": str(manifest_path),
        "output": str(output_root / "data" / "manifest_splits.csv"),
        "dry_run": args.dry_run,
        **summarize_splits(rows),
    }
    if not args.dry_run:
        write_csv_records(output_root / "data" / "manifest_splits.csv", rows)
        write_json(output_root / "reports" / "split_report.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
