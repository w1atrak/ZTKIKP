"""Central configuration: seeds, paths, crypto/ML hyperparameters."""

from pathlib import Path

# Reproducibility
SEED: int = 42

# Repository layout (paths relative to repo root)
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
RESULTS_DIR = REPO_ROOT / "results"

# TF-IDF (norm is always "l2"; see ppspam.features.make_vectorizer) and NB
# smoothing, chosen by `scripts/train.py --tune` (5-fold CV on train, refit on F1)
TFIDF_PARAMS: dict = {
    "ngram_range": (1, 1),
    "min_df": 2,
    "max_df": 1.0,
    "max_features": None,
    "sublinear_tf": False,
}
NB_ALPHA: float = 0.1

# ML / quantization (defaults; refined in later phases)
VOCAB_SIZE_CANDIDATES: tuple[int, ...] = (500, 1000, 2000)
# d = length of the encrypted vector; None keeps the full vocabulary
FEATURE_SELECTION: str = "max_features"  # see ppspam.features.SELECTION_METHODS
FEATURE_DIM: int | None = 1000  # chosen in scripts/select_features.py (A5)
SCALE_X: float = 1e4  # S_x
SCALE_W: float = 1e6  # S_w

# Paillier key sizes (bits)
KEY_SIZE_DEV: int = 1024
KEY_SIZE_PROD: int = 2048
KEY_SIZE_BENCH: int = 3072

# Train/test split
TEST_SIZE: float = 0.2
