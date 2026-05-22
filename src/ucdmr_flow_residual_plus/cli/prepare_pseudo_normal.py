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
from ucdmr_flow_residual_plus.config import load_config, resolve_dataset_root, resolve_output_root
from ucdmr_flow_residual_plus.pseudo_normal import prepare_pseudo_normal_plus


def _prepare_pseudo_worker(task: tuple[dict[str, str], Path, Path, str, float, float, float, float, bool]) -> dict[str, object]:
    row, dataset_root, output_root, method, max_outside_l1, min_quality_score, max_pseudo_blockiness, max_pseudo_chroma, skip_existing = task
    return prepare_pseudo_normal_plus(
        row,
        dataset_root=dataset_root,
        output_root=output_root,
        method=method,
        max_outside_l1=max_outside_l1,
        min_quality_score=min_quality_score,
        max_pseudo_blockiness=max_pseudo_blockiness,
        max_pseudo_chroma=max_pseudo_chroma,
        skip_existing=skip_existing,
    )


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Build plus pseudo-normal images from real crack images.")
    add_common_args(parser)
    parser.add_argument("--masks-manifest", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--method", default="opencv_telea")
    parser.add_argument("--max-outside-l1", type=float, default=1.0)
    parser.add_argument("--min-quality-score", type=float, default=0.0)
    parser.add_argument("--max-pseudo-blockiness", type=float, default=float("inf"))
    parser.add_argument("--max-pseudo-chroma", type=float, default=float("inf"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
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
    actual_workers = 0
    if not args.dry_run:
        tasks = [
            (
                row,
                dataset_root,
                output_root,
                args.method,
                args.max_outside_l1,
                args.min_quality_score,
                args.max_pseudo_blockiness,
                args.max_pseudo_chroma,
                args.skip_existing,
            )
            for row in rows
        ]
        workers = max(1, min(args.workers, len(tasks))) if tasks else 1
        if workers == 1:
            actual_workers = 1 if tasks else 0
            records = [_prepare_pseudo_worker(task) for task in tqdm(tasks, desc="plus_pseudo", unit="crack")]
        else:
            try:
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    records = list(tqdm(executor.map(_prepare_pseudo_worker, tasks, chunksize=1), total=len(tasks), desc="plus_pseudo", unit="crack"))
                actual_workers = workers
            except OSError as exc:
                print({"stage": "plus_prepare_pseudo_normal", "warning": "process_pool_unavailable_falling_back_to_single_worker", "error": str(exc)})
                actual_workers = 1
                records = [_prepare_pseudo_worker(task) for task in tqdm(tasks, desc="plus_pseudo", unit="crack")]
    summary = {
        "stage": "plus_prepare_pseudo_normal",
        "masks_manifest": str(masks_manifest),
        "output": str(output_root / "pseudo_normal" / "pseudo_manifest.csv"),
        "dry_run": args.dry_run,
        "row_count": len(rows),
        "accepted_count": sum(1 for row in records if row.get("pseudo_quality_accepted") == "1"),
        "workers": args.workers,
        "actual_workers": actual_workers,
        "skip_existing": args.skip_existing,
    }
    if not args.dry_run:
        write_csv_records(output_root / "pseudo_normal" / "pseudo_manifest.csv", records)
        write_json(output_root / "reports" / "pseudo_normal_report.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
