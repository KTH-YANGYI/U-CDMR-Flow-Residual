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
from ucdmr_flow_residual_plus.config import domain_value, load_config, resolve_dataset_root, resolve_output_root
from ucdmr_flow_residual_plus.domain_stats import write_domain_stats
from ucdmr_flow_residual_plus.manifest import limit_per_domain
from ucdmr_flow_residual_plus.mask_representations import prepare_plus_masks


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Build plus mask representations: raw, inpaint, band, gate, skeleton, SDF, thickness.")
    add_common_args(parser)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples-per-domain", type=int, default=None)
    parser.add_argument("--sdf-clip", type=float, default=64.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    dataset_root = resolve_dataset_root(config, args.dataset_root)
    output_root = resolve_output_root(config, args.output_root)
    manifest_path = args.manifest or output_root / "data" / "manifest_splits.csv"
    if args.dry_run and not Path(manifest_path).exists():
        print(
            {
                "stage": "plus_prepare_masks",
                "source_manifest": str(manifest_path),
                "output": str(output_root / "masks" / "masks_manifest.csv"),
                "dry_run": True,
                "input_exists": False,
            }
        )
        return
    rows = read_csv_records(manifest_path)
    cracks = [row for row in rows if row.get("label") == "crack" and (args.split is None or row.get("split") == args.split)]
    cracks = limit_per_domain(cracks, args.max_samples_per_domain)
    if args.max_samples is not None:
        cracks = cracks[: args.max_samples]
    records = []
    if not args.dry_run:
        for row in tqdm(cracks, desc="plus_masks", unit="crack"):
            domain = row.get("dataset_group", row.get("domain", ""))
            records.append(
                prepare_plus_masks(
                    row,
                    dataset_root=dataset_root,
                    output_root=output_root / "masks",
                    inpaint_radius=int(domain_value(config, "masks", "inpaint_radius", domain, 9)),
                    band_radius=int(domain_value(config, "masks", "band_radius", domain, 5)),
                    gate_radius=int(domain_value(config, "masks", "gate_radius", domain, 7)),
                    gate_blur=float(domain_value(config, "masks", "gate_blur", domain, 3.0)),
                    sdf_clip=args.sdf_clip,
                )
            )
    summary = {
        "stage": "plus_prepare_masks",
        "source_manifest": str(manifest_path),
        "output": str(output_root / "masks" / "masks_manifest.csv"),
        "dry_run": args.dry_run,
        "crack_count": len(cracks),
        "domains": sorted({row.get("dataset_group", row.get("domain", "")) for row in cracks}),
    }
    if not args.dry_run:
        write_csv_records(output_root / "masks" / "masks_manifest.csv", records)
        stats = write_domain_stats(records, output_root)
        summary.update(stats)
        write_json(output_root / "reports" / "mask_report.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
