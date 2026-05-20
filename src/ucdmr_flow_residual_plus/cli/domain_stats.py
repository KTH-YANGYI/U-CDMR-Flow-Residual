from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from ucdmr_flow_residual_plus.io_utils import read_csv_records, write_json

from ucdmr_flow_residual_plus.cli.common import add_common_args
from ucdmr_flow_residual_plus.config import load_config, resolve_output_root
from ucdmr_flow_residual_plus.domain_stats import write_domain_stats


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Summarize per-domain mask statistics for plus pipeline.")
    add_common_args(parser)
    parser.add_argument("--masks-manifest", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    output_root = resolve_output_root(config, args.output_root)
    masks_manifest = args.masks_manifest or output_root / "masks" / "masks_manifest.csv"
    if args.dry_run and not Path(masks_manifest).exists():
        print({"stage": "plus_domain_stats", "masks_manifest": str(masks_manifest), "dry_run": True, "input_exists": False})
        return
    rows = read_csv_records(masks_manifest)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    summary = {"stage": "plus_domain_stats", "masks_manifest": str(masks_manifest), "dry_run": args.dry_run, "row_count": len(rows)}
    if not args.dry_run:
        summary.update(write_domain_stats(rows, output_root))
        write_json(output_root / "reports" / "domain_stats_report.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
