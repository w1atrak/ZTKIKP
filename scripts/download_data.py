#!/usr/bin/env python3
"""Download SMS Spam Collection from UCI, deduplicate, split, write stats (A1)."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ppspam import data  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download raw zip")
    args = parser.parse_args()

    stats = data.prepare(force_download=args.force)
    dd = stats["deduplication"]
    print(f"raw:   {stats['raw']}")
    print(
        f"dedup: removed {dd['n_duplicates_removed']} duplicates "
        f"{dd['n_duplicates_removed_by_label']}, "
        f"conflicting labels: {dd['n_texts_with_conflicting_labels']}"
    )
    print(f"train: {stats['train']}")
    print(f"test:  {stats['test']}")
    print(f"'always ham' test accuracy: {stats['majority_baseline_accuracy_test']}")
    print(f"written: {data.TRAIN_CSV}, {data.TEST_CSV}, {data.DATA_STATS_JSON}")


if __name__ == "__main__":
    main()
