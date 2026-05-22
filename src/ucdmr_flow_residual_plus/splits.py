from __future__ import annotations

from collections import Counter, defaultdict
import random


def assign_video_splits(
    rows: list[dict[str, str]],
    *,
    seed: int = 0,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> list[dict[str, str]]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")
    videos_by_domain: dict[str, list[str]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for row in rows:
        domain = row.get("effective_domain", row.get("domain", row.get("dataset_group", "")))
        video = row.get("video_name", "")
        key = (domain, video)
        if key not in seen:
            seen.add(key)
            videos_by_domain[domain].append(video)
    split_by_key: dict[tuple[str, str], str] = {}
    for domain, videos in sorted(videos_by_domain.items()):
        rng = random.Random(f"{seed}:{domain}")
        shuffled = sorted(videos)
        rng.shuffle(shuffled)
        n = len(shuffled)
        if n == 1:
            n_train, n_val = 1, 0
        elif n == 2:
            n_train, n_val = 1, 1
        else:
            n_train = max(1, round(n * train_ratio))
            n_val = max(1, round(n * val_ratio))
            if n_train + n_val >= n:
                n_train = max(1, n - 2)
                n_val = 1
        for idx, video in enumerate(shuffled):
            split = "train" if idx < n_train else "val" if idx < n_train + n_val else "test"
            split_by_key[(domain, video)] = split
    out = []
    for row in rows:
        record = dict(row)
        domain = record.get("effective_domain", record.get("domain", record.get("dataset_group", "")))
        video = record.get("video_name", "")
        record["split"] = split_by_key[(domain, video)]
        record["split_key"] = f"{domain}::{video}"
        record["split_strategy"] = "domain_video_level"
        out.append(record)
    return out


def summarize_splits(rows: list[dict[str, str]]) -> dict[str, object]:
    split_counts = Counter(row.get("split", "") for row in rows)
    by_domain = Counter((row.get("effective_domain", row.get("domain", row.get("dataset_group", ""))), row.get("split", "")) for row in rows)
    split_keys: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split_keys[row.get("split", "")].add(row.get("split_key", ""))
    return {
        "row_count": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "by_domain_split": {f"{domain}/{split}": count for (domain, split), count in sorted(by_domain.items())},
        "split_key_counts": {split: len(keys) for split, keys in sorted(split_keys.items())},
    }
