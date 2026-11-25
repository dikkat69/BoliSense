"""
Model definitions, builders, and serialization for BoliSense.
"""

from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from src.features import TextNormalizerTransformer, LinguisticFeatureTransformer

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def build_majority_baseline() -> Pipeline:
    return Pipeline([
        ("vec", CountVectorizer()),
        ("clf", DummyClassifier(strategy="most_frequent"))
    ])


def build_naive_bayes() -> Pipeline:
    return Pipeline([
        ("vec", TfidfVectorizer(ngram_range=(1, 2), max_features=1500, sublinear_tf=True)),
        ("clf", MultinomialNB(alpha=1.0))
    ])


def build_word_logistic_regression() -> Pipeline:
    return Pipeline([
        ("vec", TfidfVectorizer(ngram_range=(1, 2), max_features=1500, sublinear_tf=True)),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42))
    ])


def build_char_logistic_regression() -> Pipeline:
    return Pipeline([
        ("vec", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=3000, sublinear_tf=True)),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42))
    ])


def build_linear_svm() -> Pipeline:
    features = FeatureUnion([
        ("w", TfidfVectorizer(ngram_range=(1, 2), max_features=1500, sublinear_tf=True)),
        ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=2500, sublinear_tf=True))
    ])
    base = LinearSVC(C=1.0, random_state=42, max_iter=2000)
    clf = CalibratedClassifierCV(estimator=base, cv=3)
    return Pipeline([("feat", features), ("clf", clf)])


def build_hybrid_model(
    apply_normalization: bool = True,
    include_linguistic: bool = True,
    include_word: bool = True,
    include_char: bool = True,
    C: float = 1.0
) -> Pipeline:
    """Build the Proposed BoliSense Hybrid Classifier."""
    feat_list = []
    if include_word:
        feat_list.append(("word", TfidfVectorizer(ngram_range=(1, 2), max_features=1500, sublinear_tf=True)))
    if include_char:
        feat_list.append(("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=3000, sublinear_tf=True)))
    if include_linguistic:
        feat_list.append(("ling", LinguisticFeatureTransformer()))

    steps = []
    if apply_normalization:
        steps.append(("norm", TextNormalizerTransformer(True)))
    steps.append(("feat", FeatureUnion(feat_list)))
    steps.append(("clf", LogisticRegression(C=C, max_iter=1000, random_state=42)))
    return Pipeline(steps)


def save_model(model: Pipeline, filename: str) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = MODELS_DIR / filename
    joblib.dump(model, target)
    return target


def load_model(filename: str) -> Pipeline:
    target = MODELS_DIR / filename
    if not target.exists():
        # Fallback to subdirectories if not migrated yet
        alt_targets = [
            MODELS_DIR / "hybrid" / filename,
            MODELS_DIR / "classical" / filename
        ]
        for at in alt_targets:
            if at.exists():
                return joblib.load(at)
        raise FileNotFoundError(f"Model file not found: {target}")
    return joblib.load(target)
