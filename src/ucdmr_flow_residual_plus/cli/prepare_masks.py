from __future__ import annotations

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **_: object):
        return iterable

from ucdmr_flow_residual_plus.io_utils import read_csv_records, write_csv_records, write_json

from ucdmr_flow_residual_plus.cli.common import add_common_args
from ucdmr_flow_residual_plus.config import domain_value, load_config, resolve_dataset_root, resolve_output_root
from ucdmr_flow_residual_plus.constants import base_domain_name, mask_domain_from_row
from ucdmr_flow_residual_plus.domain_stats import write_domain_stats
from ucdmr_flow_residual_plus.manifest import limit_per_domain
from ucdmr_flow_residual_plus.mask_representations import prepare_plus_masks


def _prepare_mask_worker(task: tuple[dict[str, str], Path, Path, int, int, int, float, float, bool]) -> dict[str, object]:
    row, dataset_root, output_root, inpaint_radius, band_radius, gate_radius, gate_blur, sdf_clip, skip_existing = task
    return prepare_plus_masks(
        row,
        dataset_root=dataset_root,
        output_root=output_root,
        inpaint_radius=inpaint_radius,
        band_radius=band_radius,
        gate_radius=gate_radius,
        gate_blur=gate_blur,
        sdf_clip=sdf_clip,
        skip_existing=skip_existing,
    )


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Build plus mask representations: raw, inpaint, band, gate, skeleton, SDF, thickness.")
    add_common_args(parser)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples-per-domain", type=int, default=None)
    parser.add_argument("--sdf-clip", type=float, default=64.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
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
    actual_workers = 0
    if not args.dry_run:
        tasks = []
        for row in cracks:
            domain = mask_domain_from_row(row)
            config_domain = base_domain_name(domain)
            tasks.append(
                (
                    row,
                    dataset_root,
                    output_root / "masks",
                    int(domain_value(config, "masks", "inpaint_radius", config_domain, 9)),
                    int(domain_value(config, "masks", "band_radius", config_domain, 5)),
                    int(domain_value(config, "masks", "gate_radius", config_domain, 7)),
                    float(domain_value(config, "masks", "gate_blur", config_domain, 3.0)),
                    args.sdf_clip,
                    args.skip_existing,
                )
            )
        workers = max(1, min(args.workers, len(tasks))) if tasks else 1
        if workers == 1:
            actual_workers = 1 if tasks else 0
            records = [_prepare_mask_worker(task) for task in tqdm(tasks, desc="plus_masks", unit="crack")]
        else:
            try:
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    records = list(tqdm(executor.map(_prepare_mask_worker, tasks, chunksize=1), total=len(tasks), desc="plus_masks", unit="crack"))
                actual_workers = workers
            except OSError as exc:
                print({"stage": "plus_prepare_masks", "warning": "process_pool_unavailable_falling_back_to_single_worker", "error": str(exc)})
                actual_workers = 1
                records = [_prepare_mask_worker(task) for task in tqdm(tasks, desc="plus_masks", unit="crack")]
    summary = {
        "stage": "plus_prepare_masks",
        "source_manifest": str(manifest_path),
        "output": str(output_root / "masks" / "masks_manifest.csv"),
        "dry_run": args.dry_run,
        "crack_count": len(cracks),
        "domains": sorted({mask_domain_from_row(row) for row in cracks}),
        "workers": args.workers,
        "actual_workers": actual_workers,
        "skip_existing": args.skip_existing,
    }
    if not args.dry_run:
        write_csv_records(output_root / "masks" / "masks_manifest.csv", records)
        stats = write_domain_stats(records, output_root)
        summary.update(stats)
        write_json(output_root / "reports" / "mask_report.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
