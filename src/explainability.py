"""
Unified inference and feature contribution explainability for BoliSense.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from src.models import load_model, MODELS_DIR
from src.features import DialectNormalizer, LinguisticFeatureExtractor


class BoliSenseInferenceEngine:
    def __init__(self):
        self.normalizer = DialectNormalizer()
        self.extractor = LinguisticFeatureExtractor()
        self.models = {}
        self._load_models()

    def _load_models(self):
        model_files = [
            ("Proposed BoliSense Hybrid", "bolisense_hybrid.joblib"),
            ("Linear SVM (Word+Char)", "linear_svm.joblib"),
            ("Logistic Regression (Word)", "word_logistic_regression.joblib"),
            ("Logistic Regression (Char)", "char_logistic_regression.joblib"),
            ("Multinomial Naive Bayes", "naive_bayes.joblib"),
            ("Majority Baseline", "majority_baseline.joblib")
        ]
        for name, fn in model_files:
            try:
                self.models[name] = load_model(fn)
            except Exception:
                pass

    def predict(self, text: str, model_name: str = "Proposed BoliSense Hybrid") -> dict:
        if not text or not text.strip():
            return {"sentiment": "Unknown", "confidence": 0.0, "error": "Empty input"}

        model = self.models.get(model_name)
        if not model:
            model = list(self.models.values())[0] if self.models else None
            model_name = list(self.models.keys())[0] if self.models else "None"

        if not model:
            return {"sentiment": "Unavailable", "confidence": 0.0, "error": "No models loaded"}

        pred = int(model.predict([text])[0])
        sentiment = "Positive" if pred == 1 else "Negative"

        confidence = 0.5
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([text])[0]
            confidence = float(probs[pred])
        elif hasattr(model, "decision_function"):
            score = float(model.decision_function([text])[0])
            confidence = float(1.0 / (1.0 + np.exp(-abs(score))))

        explanation = self.explain(text, model)

        return {
            "sentence": text,
            "normalized": self.normalizer.normalize(text),
            "model": model_name,
            "label": pred,
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
            "explanation": explanation
        }

    def explain(self, text: str, model=None) -> dict:
        """Extract linguistic cues and local linear feature contributions."""
        cues = {
            "negation": self.extractor.detect_negation(text),
            "contrast": self.extractor.detect_contrast(text),
            "lexicon": self.extractor.score_lexicon(text)
        }
        contribs = {"top_positive": [], "top_negative": [], "net_logit": 0.0}

        if model and hasattr(model, "named_steps") and "clf" in model.named_steps:
            clf = model.named_steps["clf"]
            feat_step = model.named_steps.get("feat")
            norm_step = model.named_steps.get("norm")

            if feat_step and hasattr(clf, "coef_"):
                norm_text = norm_step.transform([text])[0] if norm_step else text
                X_sparse = feat_step.transform([norm_text])
                feature_names = feat_step.get_feature_names_out()
                weights = clf.coef_[0]
                intercept = float(clf.intercept_[0])

                X_dense = X_sparse.toarray()[0]
                local_c = X_dense * weights
                active_idx = np.where(np.abs(local_c) > 1e-5)[0]

                active = []
                for idx in active_idx:
                    fn = feature_names[idx].replace("feat__", "").replace("word__", "").replace("char__", "").replace("ling__", "")
                    active.append({
                        "feature": fn,
                        "contribution": round(float(local_c[idx]), 4)
                    })
                act_df = pd.DataFrame(active)
                if not act_df.empty:
                    pos = act_df[act_df["contribution"] > 0].sort_values(by="contribution", ascending=False).head(5)
                    neg = act_df[act_df["contribution"] < 0].sort_values(by="contribution", ascending=True).head(5)
                    contribs["top_positive"] = pos.to_dict(orient="records")
                    contribs["top_negative"] = neg.to_dict(orient="records")
                contribs["net_logit"] = round(intercept + float(np.sum(local_c)), 4)

        return {"cues": cues, "contributions": contribs}

    def predict_and_explain(self, text: str, model_name: str = "Proposed BoliSense Hybrid") -> dict:
        """Convenience method returning flattened explanation structure."""
        res = self.predict(text, model_name=model_name)
        exp = res.get("explanation", {})
        contribs = exp.get("contributions", {})
        return {
            "sentence": res["sentence"],
            "normalized": res["normalized"],
            "prediction_label": res["sentiment"],
            "confidence": res["confidence"],
            "net_logit": contribs.get("net_logit", 0.0),
            "positive_contributors": contribs.get("top_positive", []),
            "negative_contributors": contribs.get("top_negative", []),
            "cues": exp.get("cues", {}),
            "raw": res
        }

    def predict_all(self, text: str) -> list[dict]:
        results = []
        for name in self.models.keys():
            res = self.predict(text, model_name=name)
            results.append({
                "Model": name,
                "Sentiment": res["sentiment"],
                "Confidence": f"{res['confidence']*100:.1f}%"
            })
        return results
