"""Deduplication and split (phase A1)."""

import pandas as pd
import pytest

from ppspam import data


def _frame() -> pd.DataFrame:
    rows = [("ham", f"hello {i}") for i in range(40)]
    rows += [("spam", f"win prize {i}") for i in range(10)]
    rows += [("ham", "hello 0"), ("spam", "win prize 0"), ("spam", "win prize 0")]
    return pd.DataFrame(rows, columns=["label", "text"])


def test_deduplicate_counts():
    out, report = data.deduplicate(_frame())
    assert len(out) == 50
    assert out["text"].is_unique
    assert report["n_duplicates_removed"] == 3
    assert report["n_duplicates_removed_by_label"] == {"spam": 2, "ham": 1}


def test_split_is_stratified_and_disjoint():
    dedup, _ = data.deduplicate(_frame())
    train, test = data.split(dedup, test_size=0.2, seed=0)
    assert len(train) + len(test) == len(dedup)
    assert set(train["id"]).isdisjoint(test["id"])
    assert (test["label"] == "spam").sum() == 2


@pytest.mark.skipif(not data.TEST_CSV.exists(), reason="run scripts/download_data.py")
def test_processed_split_has_no_overlap():
    train, test = data.load_split()
    assert set(train["text"]).isdisjoint(test["text"])
    assert set(train["label"]) == set(test["label"]) == set(data.LABELS)
