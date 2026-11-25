"""
BoliSense Training Script.
Prepares datasets, trains all classical models and the Proposed Hybrid model,
and serializes artifacts to models/.
"""

from pathlib import Path
import pandas as pd
from src.dataset import load_raw_dataset, validate_and_clean_dataset, partition_dataset, PROCESSED_DATA_DIR
from src.models import (
    build_majority_baseline,
    build_naive_bayes,
    build_word_logistic_regression,
    build_char_logistic_regression,
    build_linear_svm,
    build_hybrid_model,
    save_model,
    MODELS_DIR
)


def train_all_models():
    print("=" * 60)
    print("      BoliSense Model Training Pipeline")
    print("=" * 60)

    # 1. Dataset verification & splitting
    print("\n[1/3] Loading raw dataset and preparing splits...")
    raw_df = load_raw_dataset()
    cleaned_df, audit = validate_and_clean_dataset(raw_df)
    train_df, val_df, test_df = partition_dataset(cleaned_df, seed=42)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DATA_DIR / "validation.csv", index=False)
    test_df.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)

    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    X_train, y_train = train_df["sentence"], train_df["label"]

    # 2. Build & train model suite
    models = {
        "majority_baseline.joblib": ("Majority Baseline", build_majority_baseline()),
        "naive_bayes.joblib": ("Multinomial Naive Bayes", build_naive_bayes()),
        "word_logistic_regression.joblib": ("Word TF-IDF + Logistic Reg.", build_word_logistic_regression()),
        "char_logistic_regression.joblib": ("Char TF-IDF + Logistic Reg.", build_char_logistic_regression()),
        "linear_svm.joblib": ("TF-IDF + Linear SVM", build_linear_svm()),
        "bolisense_hybrid.joblib": ("Proposed BoliSense Hybrid", build_hybrid_model()),
    }

    print("\n[2/3] Training and serializing models...")
    for filename, (name, pipe) in models.items():
        pipe.fit(X_train, y_train)
        saved_path = save_model(pipe, filename)
        print(f"  Trained {name:30} -> Saved to {saved_path.name}")

    print("\n[3/3] Training complete. All models saved directly to models/.")
    print("=" * 60)


if __name__ == "__main__":
    train_all_models()
