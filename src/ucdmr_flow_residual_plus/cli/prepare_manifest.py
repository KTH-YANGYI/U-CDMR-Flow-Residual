from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from ucdmr_flow_residual_plus.io_utils import write_csv_records, write_json

from ucdmr_flow_residual_plus.cli.common import add_common_args
from ucdmr_flow_residual_plus.config import load_config, nested_get, resolve_dataset_root, resolve_output_root
from ucdmr_flow_residual_plus.manifest import filter_valid_rows, limit_per_domain, load_merged_manifest, summarize


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Prepare U-CDMR-Flow-Residual+ crack/normal manifest; broken is always ignored.")
    add_common_args(parser)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--max-samples-per-domain", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    dataset_root = resolve_dataset_root(config, args.dataset_root)
    output_root = resolve_output_root(config, args.output_root)
    manifest_value = args.manifest or nested_get(config, ("dataset", "manifest"), "manifest_merged.csv")
    manifest_path = Path(manifest_value)
    if not manifest_path.is_absolute():
        manifest_path = dataset_root / manifest_path
    rows = load_merged_manifest(manifest_path)
    filtered = filter_valid_rows(rows, max_samples=args.max_samples)
    filtered = limit_per_domain(filtered, args.max_samples_per_domain)
    summary = {
        "stage": "plus_prepare_manifest",
        "source_manifest": str(manifest_path),
        "output": str(output_root / "data" / "manifest_filtered.csv"),
        "dry_run": args.dry_run,
        **summarize(filtered),
    }
    if not args.dry_run:
        write_csv_records(output_root / "data" / "manifest_filtered.csv", filtered)
        write_json(output_root / "reports" / "manifest_report.json", summary)
    print(summary)


if __name__ == "__main__":
    main()

