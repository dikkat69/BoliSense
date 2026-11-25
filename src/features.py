"""
Text preprocessing, dialect normalization, and linguistic feature extraction for BoliSense.
"""

import re
import json
import unicodedata
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

RESOURCES_DIR = Path(__file__).resolve().parent.parent / "data" / "resources"


def clean_text(text: str) -> str:
    """Normalize unicode and whitespace."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def word_tokenize(text: str) -> list[str]:
    """Tokenize into alphanumeric words and punctuation."""
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)


def char_ngrams(text: str, n_range: tuple[int, int] = (3, 5)) -> list[str]:
    """Extract character n-grams."""
    clean = f" {text.strip().lower()} "
    ngrams = []
    for n in range(n_range[0], n_range[1] + 1):
        for i in range(len(clean) - n + 1):
            ngrams.append(clean[i:i + n])
    return ngrams


class DialectNormalizer:
    """Normalizes informal Roman-script Haryanvi tokens to canonical forms."""
    def __init__(self, dict_path: Path = RESOURCES_DIR / "normalization.json"):
        self.mappings = {}
        if dict_path.exists():
            with open(dict_path, "r", encoding="utf-8") as f:
                self.mappings = json.load(f).get("mappings", {})

    def normalize(self, text: str) -> str:
        words = text.split()
        norm_words = []
        for w in words:
            match = re.match(r"^(\W*)([\w\-]+)(\W*)$", w, re.UNICODE)
            if match:
                prefix, token, suffix = match.groups()
                norm_words.append(f"{prefix}{self.mappings.get(token.lower(), token)}{suffix}")
            else:
                norm_words.append(w)
        return " ".join(norm_words)

    def get_replaced_tokens(self, text: str) -> list[tuple[str, str]]:
        replaced = []
        for w in text.split():
            match = re.match(r"^(\W*)([\w\-]+)(\W*)$", w, re.UNICODE)
            if match:
                _, token, _ = match.groups()
                norm = self.mappings.get(token.lower(), token)
                if token.lower() != norm.lower():
                    replaced.append((token, norm))
        return replaced


class LinguisticFeatureExtractor:
    """Extracts negation scope, contrastive clauses, and sentiment lexicon cues."""
    def __init__(self, lexicon_path: Path = RESOURCES_DIR / "sentiment_lexicon.json"):
        self.lexicon = {}
        if lexicon_path.exists():
            with open(lexicon_path, "r", encoding="utf-8") as f:
                self.lexicon = json.load(f)
        self.pos_words = set(self.lexicon.get("positive", []))
        self.neg_words = set(self.lexicon.get("negative", []))
        self.neg_markers = set(self.lexicon.get("negation_markers", []))
        self.con_markers = set(self.lexicon.get("contrast_markers", []))

    def detect_negation(self, text: str) -> dict:
        words = [w.lower().strip(".,!?") for w in text.split()]
        found = [w for w in words if w in self.neg_markers]
        neg_pos, neg_neg = 0, 0
        for i, w in enumerate(words):
            if w in self.neg_markers:
                window = words[max(0, i - 2):min(len(words), i + 3)]
                for neighbor in window:
                    if neighbor in self.pos_words:
                        neg_pos += 1
                    elif neighbor in self.neg_words:
                        neg_neg += 1
        return {
            "has_negation": int(len(found) > 0),
            "negation_count": len(found),
            "negation_tokens": found,
            "negates_positive": neg_pos,
            "negates_negative": neg_neg,
        }

    def detect_contrast(self, text: str) -> dict:
        pattern = r"\b(" + "|".join(re.escape(m) for m in self.con_markers) + r")\b"
        match = re.search(pattern, text.lower())
        if not match:
            return {"has_contrast": 0, "contrast_marker": None, "clause_1": text.strip(), "clause_2": "", "clause_2_positive_score": 0, "clause_2_negative_score": 0}
        m = match.group(1)
        c1 = text[:match.start()].strip()
        c2 = text[match.end():].strip()
        c2_words = [w.lower().strip(".,!?") for w in c2.split()]
        return {
            "has_contrast": 1,
            "contrast_marker": m,
            "clause_1": c1,
            "clause_2": c2,
            "clause_2_positive_score": sum(1 for w in c2_words if w in self.pos_words),
            "clause_2_negative_score": sum(1 for w in c2_words if w in self.neg_words),
        }

    def score_lexicon(self, text: str) -> dict:
        words = [w.lower().strip(".,!?") for w in text.split()]
        pos = [w for w in words if w in self.pos_words]
        neg = [w for w in words if w in self.neg_words]
        return {
            "positive_hits": pos,
            "negative_hits": neg,
            "positive_count": len(pos),
            "negative_count": len(neg),
            "balance": len(pos) - len(neg),
        }

    def extract_feature_vector(self, text: str) -> np.ndarray:
        words = text.split()
        n = self.detect_negation(text)
        c = self.detect_contrast(text)
        l = self.score_lexicon(text)
        return np.array([
            len(words), len(text),
            n["has_negation"], n["negation_count"], n["negates_positive"], n["negates_negative"],
            c["has_contrast"], c["clause_2_positive_score"], c["clause_2_negative_score"],
            l["positive_count"], l["negative_count"], l["balance"],
            text.count("!"), text.count("?")
        ], dtype=np.float32)

    @property
    def feature_names(self) -> list[str]:
        return [
            "ling_word_count", "ling_char_count", "ling_has_negation", "ling_negation_count",
            "ling_negates_positive", "ling_negates_negative", "ling_has_contrast",
            "ling_clause2_pos_score", "ling_clause2_neg_score", "ling_pos_lexicon_count",
            "ling_neg_lexicon_count", "ling_lexicon_balance", "ling_exclamation_count", "ling_question_count"
        ]


class TextNormalizerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, apply: bool = True):
        self.apply = apply
        self.normalizer = DialectNormalizer()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self.normalizer.normalize(str(t)) for t in X] if self.apply else list(X)


class LinguisticFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.extractor = LinguisticFeatureExtractor()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = [self.extractor.extract_feature_vector(str(t)) for t in X]
        return sp.csr_matrix(np.array(features, dtype=np.float32))

    def get_feature_names_out(self, input_features=None):
        return np.array(self.extractor.feature_names)
