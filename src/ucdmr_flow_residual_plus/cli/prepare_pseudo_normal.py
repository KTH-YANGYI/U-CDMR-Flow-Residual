from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **_: object):
        return iterable

from ucdmr_flow_residual_plus.io_utils import read_csv_records, write_csv_records, write_json

from ucdmr_flow_residual_plus.cli.common import add_common_args
from ucdmr_flow_residual_plus.config import load_config, resolve_dataset_root, resolve_output_root
from ucdmr_flow_residual_plus.pseudo_normal import prepare_pseudo_normal_plus


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Build plus pseudo-normal images from real crack images.")
    add_common_args(parser)
    parser.add_argument("--masks-manifest", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--method", default="opencv_telea")
    parser.add_argument("--max-outside-l1", type=float, default=1.0)
    parser.add_argument("--min-quality-score", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    dataset_root = resolve_dataset_root(config, args.dataset_root)
    output_root = resolve_output_root(config, args.output_root)
    masks_manifest = args.masks_manifest or output_root / "masks" / "masks_manifest.csv"
    if args.dry_run and not Path(masks_manifest).exists():
        print(
            {
                "stage": "plus_prepare_pseudo_normal",
                "masks_manifest": str(masks_manifest),
                "output": str(output_root / "pseudo_normal" / "pseudo_manifest.csv"),
                "dry_run": True,
                "input_exists": False,
            }
        )
        return
    rows = read_csv_records(masks_manifest)
    rows = [row for row in rows if args.split is None or row.get("split") == args.split]
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    records = []
    if not args.dry_run:
        records = [
            prepare_pseudo_normal_plus(
                row,
                dataset_root=dataset_root,
                output_root=output_root,
                method=args.method,
                max_outside_l1=args.max_outside_l1,
                min_quality_score=args.min_quality_score,
            )
            for row in tqdm(rows, desc="plus_pseudo", unit="crack")
        ]
    summary = {
        "stage": "plus_prepare_pseudo_normal",
        "masks_manifest": str(masks_manifest),
        "output": str(output_root / "pseudo_normal" / "pseudo_manifest.csv"),
        "dry_run": args.dry_run,
        "row_count": len(rows),
        "accepted_count": sum(1 for row in records if row.get("pseudo_quality_accepted") == "1"),
    }
    if not args.dry_run:
        write_csv_records(output_root / "pseudo_normal" / "pseudo_manifest.csv", records)
        write_json(output_root / "reports" / "pseudo_normal_report.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
