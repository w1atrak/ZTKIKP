"""Dataset download, load, deduplication, and train/test split (phase A1)."""

from __future__ import annotations

import csv
import io
import json
import re
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ppspam import config

UCI_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
RAW_FILENAME = "SMSSpamCollection"
LABELS = ("ham", "spam")

TRAIN_CSV = config.DATA_PROCESSED / "train.csv"
TEST_CSV = config.DATA_PROCESSED / "test.csv"
DATA_STATS_JSON = config.RESULTS_DIR / "data_stats.json"


def download_raw(dest_dir: Path = config.DATA_RAW, force: bool = False) -> Path:
    """Download the UCI zip and extract ``SMSSpamCollection`` into ``dest_dir``."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / RAW_FILENAME
    if target.exists() and not force:
        return target
    with urllib.request.urlopen(UCI_URL, timeout=60) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extract(RAW_FILENAME, dest_dir)
    return target


def load_raw(path: Path = config.DATA_RAW / RAW_FILENAME) -> pd.DataFrame:
    """Load the tab-separated raw file into a ``label``/``text`` frame."""
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["label", "text"],
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
        dtype=str,
        keep_default_na=False,
    )
    unknown = set(df["label"]) - set(LABELS)
    if unknown:
        raise ValueError(f"Unexpected labels in raw data: {unknown}")
    return df


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop exact duplicate messages (D9); returns the frame and a short report.

    A text that appears with both labels would be ambiguous; it is reported and
    only its first occurrence is kept.
    """
    conflicting = df.groupby("text")["label"].nunique()
    n_conflicting = int((conflicting > 1).sum())
    dup_mask = df.duplicated(subset="text", keep="first")
    report = {
        "n_raw": len(df),
        "n_duplicates_removed": int(dup_mask.sum()),
        "n_duplicates_removed_by_label": {
            k: int(v) for k, v in df.loc[dup_mask, "label"].value_counts().items()
        },
        "n_texts_with_conflicting_labels": n_conflicting,
    }
    out = df.loc[~dup_mask].reset_index(drop=True)
    report["n_dedup"] = len(out)
    return out, report


def split(
    df: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    seed: int = config.SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split; keeps the original row id as ``id``."""
    df = df.reset_index(drop=True).rename_axis("id").reset_index()
    train, test = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=seed
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def load_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the frozen processed split written by :func:`prepare`."""
    read = dict(dtype={"id": int, "label": str, "text": str}, keep_default_na=False)
    return pd.read_csv(TRAIN_CSV, **read), pd.read_csv(TEST_CSV, **read)


def _class_counts(df: pd.DataFrame) -> dict:
    counts = df["label"].value_counts()
    total = len(df)
    return {
        "total": total,
        **{lab: int(counts.get(lab, 0)) for lab in LABELS},
        "spam_ratio": round(float(counts.get("spam", 0)) / total, 4),
    }


def _length_stats(df: pd.DataFrame) -> dict:
    out = {}
    for lab in LABELS:
        chars = df.loc[df["label"] == lab, "text"].str.len()
        words = df.loc[df["label"] == lab, "text"].str.split().str.len()
        out[lab] = {
            "chars_mean": round(float(chars.mean()), 2),
            "chars_median": float(chars.median()),
            "chars_max": int(chars.max()),
            "words_mean": round(float(words.mean()), 2),
            "words_median": float(words.median()),
        }
    return out


_WORD_RE = re.compile(r"[a-z']{3,}")
_STOP = frozenset(
    "the and you for your are have that this with not but can will get all "
    "just now what when how out its it's was then there his her him she they "
    "our from has been i'm i'll don't".split()
)


def _top_words(df: pd.DataFrame, k: int = 20) -> dict:
    out = {}
    for lab in LABELS:
        counter: Counter[str] = Counter()
        for text in df.loc[df["label"] == lab, "text"]:
            counter.update(w for w in _WORD_RE.findall(text.lower()) if w not in _STOP)
        out[lab] = counter.most_common(k)
    return out


def dataset_stats(
    raw: pd.DataFrame, dedup_report: dict, train: pd.DataFrame, test: pd.DataFrame
) -> dict:
    """Counts and a short EDA summary for the report (written to data_stats.json)."""
    dedup = pd.concat([train, test])
    return {
        "source": UCI_URL,
        "seed": config.SEED,
        "test_size": config.TEST_SIZE,
        "raw": _class_counts(raw),
        "deduplication": dedup_report,
        "dedup": _class_counts(dedup),
        "train": _class_counts(train),
        "test": _class_counts(test),
        "majority_baseline_accuracy_test": round(
            float((test["label"] == "ham").mean()), 4
        ),
        "message_length": _length_stats(dedup),
        "top_words": _top_words(train),
    }


def prepare(force_download: bool = False) -> dict:
    """Full A1 pipeline: download → load → dedup → split → write CSVs + stats."""
    raw_path = download_raw(force=force_download)
    raw = load_raw(raw_path)
    dedup, report = deduplicate(raw)
    train, test = split(dedup)

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_CSV, index=False)
    test.to_csv(TEST_CSV, index=False)

    stats = dataset_stats(raw, report, train, test)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_STATS_JSON.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    return stats
