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

# TF-IDF (norm is always "l2"; see ppspam.features.make_vectorizer)
TFIDF_PARAMS: dict = {
    "ngram_range": (1, 1),
    "min_df": 1,
    "max_df": 1.0,
    "max_features": None,
    "sublinear_tf": True,
}

# ML / quantization (defaults; refined in later phases)
VOCAB_SIZE_CANDIDATES: tuple[int, ...] = (500, 1000, 2000)
FEATURE_DIM: int = 1000  # chosen after experiment E1; placeholder for phase 0
SCALE_X: float = 1e4  # S_x
SCALE_W: float = 1e6  # S_w

# Paillier key sizes (bits)
KEY_SIZE_DEV: int = 1024
KEY_SIZE_PROD: int = 2048
KEY_SIZE_BENCH: int = 3072

# Train/test split
TEST_SIZE: float = 0.2
