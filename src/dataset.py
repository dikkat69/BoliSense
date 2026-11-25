"""
BoliSense dataset loading, cleaning, and partitioning module.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "haryanvi_sentiment.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHALLENGE_DATA_PATH = DATA_DIR / "challenge" / "linguistic_challenge.csv"
RESOURCES_DIR = DATA_DIR / "resources"


def load_raw_dataset(filepath: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw dataset CSV."""
    path = Path(filepath)
    if not path.exists():
        # Fallback to legacy path if not yet renamed
        alt_path = DATA_DIR / "raw" / "haryanvi_combined.csv"
        if alt_path.exists():
            return pd.read_csv(alt_path)
        raise FileNotFoundError(f"Raw dataset not found at: {path}")
    return pd.read_csv(path)


def validate_and_clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Validate dataset integrity and remove exact duplicate rows."""
    cleaned_df = df.dropna().copy()
    cleaned_df.columns = [c.strip().lower() for c in cleaned_df.columns]
    cleaned_df["sentence"] = cleaned_df["sentence"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    cleaned_df = cleaned_df[cleaned_df["sentence"].str.len() > 0].copy()

    if "type" in cleaned_df.columns:
        cleaned_df["type"] = cleaned_df["type"].astype(str).str.strip().str.lower().replace({"codemixed": "code-mixed"})
    else:
        cleaned_df["type"] = "general"

    cleaned_df["label"] = cleaned_df["label"].astype(int)
    cleaned_df = cleaned_df.drop_duplicates(subset=["sentence", "label"]).copy()

    audit = {
        "raw_rows": len(df),
        "clean_rows": len(cleaned_df),
        "labels": cleaned_df["label"].value_counts().to_dict(),
        "categories": cleaned_df["type"].value_counts().to_dict()
    }
    return cleaned_df, audit


def partition_dataset(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reproducible 70-15-15 stratified partition."""
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label"], random_state=seed)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def load_processed_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load saved train, validation, and test splits."""
    return (
        pd.read_csv(PROCESSED_DATA_DIR / "train.csv"),
        pd.read_csv(PROCESSED_DATA_DIR / "validation.csv"),
        pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    )


# Convenience alias
load_splits = load_processed_splits



def load_challenge_dataset() -> pd.DataFrame:
    """Load frozen linguistic challenge dataset."""
    return pd.read_csv(CHALLENGE_DATA_PATH)
