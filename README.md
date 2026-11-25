# BoliSense: Dialect-Aware Sentiment Analysis for Haryanvi Text

> **Tagline**: *Understanding Regional Language Beyond Keywords*<br>
> **Course**: Natural Language Processing (CSE Final Year)<br>
> **Primary Approach**: Classical NLP & Linguistic Feature Engineering<br>
> **Proposed Model**: BoliSense Hybrid Classifier

---

## 1. Project Overview

**BoliSense** is an academic Natural Language Processing (NLP) framework developed to classify sentiment in **Roman-script Haryanvi**, an extreme low-resource regional Indo-Aryan dialect spoken by over 10 million people in northern India.

Unlike standardized languages, colloquial Romanized Haryanvi lacks standardized orthography, uses dialectal negation terms (*konya*, *koni*, *na*), and is heavily code-mixed with conversational Hindi and English. Large general-purpose language models often suffer from substantial subword token fragmentation on non-standardized phonetic dialects and risk overfitting on small regional corpora.

BoliSense develops an interpretable, lightweight **Dialect-Aware Hybrid Classifier** combining:
- Whole-token phonetic normalization (e.g., *badhiya* $\rightarrow$ *badiya*, *ghna* $\rightarrow$ *ghana*).
- Multi-scale character n-grams (3–5 characters) for spelling-variant resilience.
- Word n-grams (1–2 words) with sublinear term-frequency scaling.
- Explicit domain linguistic features: windowed negation scope extraction, contrastive clause segmentation (*par*, *lekin*), and regional sentiment lexicon balances.

---

## 2. Experimental Results Summary

### 2.1 Model Selection on Validation Set ($N=82$)
All hyperparameter tuning and model selection were performed strictly on the validation partition:
- **Majority Baseline**: 52.44% Accuracy | 0.3440 Macro F1
- **Multinomial Naive Bayes**: 76.83% Accuracy | 0.7655 Macro F1
- **Word TF-IDF + Logistic Reg.**: 76.83% Accuracy | 0.7655 Macro F1
- **Char TF-IDF + Logistic Reg.**: 74.39% Accuracy | 0.7429 Macro F1
- **TF-IDF + Linear SVM**: 73.17% Accuracy | 0.7315 Macro F1
- **Proposed BoliSense Hybrid**: **81.71% Accuracy | 0.8157 Macro F1** (Selected for deployment)

### 2.2 Independent Held-Out Test Set Performance ($N=82$)

| Model Architecture | Accuracy | Macro Precision | Macro Recall | Macro F1 | Positive F1 | Interactive Latency | Batched Latency | Storage Footprint |
|---|---|---|---|---|---|---|---|---|
| **Majority Baseline** | 53.66% | 0.2683 | 0.5000 | 0.3492 | 0.0000 | 0.08 ms | 0.007 ms | 7.1 KB |
| **Multinomial Naive Bayes** | 79.27% | 0.8017 | 0.7853 | 0.7873 | 0.7536 | 0.42 ms | 0.016 ms | 106.8 KB |
| **Word TF-IDF + Logistic Reg.** | 81.71% | 0.8185 | 0.8134 | 0.8148 | 0.7945 | 0.43 ms | 0.015 ms | 70.9 KB |
| **Char TF-IDF + Logistic Reg.** | 81.71% | 0.8222 | 0.8116 | 0.8137 | 0.7887 | 0.44 ms | 0.043 ms | 127.6 KB |
| **TF-IDF + Linear SVM** | 82.93% | 0.8286 | 0.8301 | 0.8289 | 0.8205 | 2.32 ms | 0.078 ms | 241.6 KB |
| **Proposed BoliSense Hybrid** | **89.02%** | **0.8904** | **0.8888** | **0.8894** | **0.8800** | **1.22 ms** | **0.074 ms** | **199.8 KB** |

*Confusion matrix for Proposed Hybrid ($N=82$): $\text{TN}=40, \text{FP}=4, \text{FN}=5, \text{TP}=33$.*

### 2.3 Additional Validations
- **Grouped-by-Template 5-Fold Cross-Validation**: **83.14% Mean Accuracy ($\pm 4.65\%$)**, **0.8299 Mean Macro F1 ($\pm 0.0458$)**, evaluating generalization across distinct syntactic templates.
- **Additive Feature Ablation (Macro F1)**:
  - Combined Word + Char TF-IDF: 0.8030
  - Combined + Dialect Normalization: 0.8157 (+1.27 pp)
  - Full BoliSense Hybrid (+ Lexicon + Negation + Contrast): **0.8894 (+8.64 pp)**
- **Linguistic Challenge Dataset ($N=65$)**: **67.69% Accuracy** across 32 minimal pairs testing adversarial negation and polarity-flip structures.
  - Spelling Invariance Consistency: **100.0%**
  - Spelling Correctness: **66.7%**
  - Polarity-Flip Sensitivity: **37.9%**

---

## 3. Repository Structure

```text
BoliSense/
├── app.py                      # Interactive Streamlit research dashboard
├── train.py                    # End-to-end dataset preparation & model training
├── evaluate.py                 # Comprehensive benchmark & figure generation
├── predict.py                  # CLI inference with linguistic cue explainability
├── requirements.txt            # Minimal classical NLP dependencies
├── LICENSE                     # MIT License
├── README.md                   # Project overview & documentation
├── data/
│   ├── challenge/
│   │   └── linguistic_challenge.csv # 65 adversarial stress-test samples
│   ├── processed/
│   │   ├── train.csv                # 70% split (382 samples)
│   │   ├── validation.csv           # 15% split (82 samples)
│   │   └── test.csv                 # 15% split (82 samples)
│   ├── raw/
│   │   └── haryanvi_sentiment.csv   # Complete raw dataset (647 samples)
│   └── resources/
│       ├── normalization.json       # Dialect phonetic normalization map
│       └── sentiment_lexicon.json   # Regional polarity lexicon
├── models/                     # Serialized production models (< 1 MB total)
│   ├── bolisense_hybrid.joblib
│   ├── char_logistic_regression.joblib
│   ├── linear_svm.joblib
│   ├── majority_baseline.joblib
│   ├── naive_bayes.joblib
│   └── word_logistic_regression.joblib
├── notebooks/
│   └── BoliSense.ipynb         # Standalone experimental notebook
├── report/
│   ├── Project_Report.md       # Comprehensive academic report
│   ├── Project_Report.pdf      # Publication-ready PDF report with diagrams
│   └── figures/                # Embedded report figures
├── results/                    # Generated evaluation metrics and plots
│   ├── figures/
│   │   ├── ablation_study.png
│   │   ├── confusion_matrix.png
│   │   └── model_comparison.png
│   ├── ablation_results.csv
│   ├── category_results.csv
│   ├── challenge_results.csv
│   ├── cross_validation.csv
│   ├── minimal_pair_results.csv
│   ├── model_comparison.csv
│   ├── test_predictions.csv
│   └── validation_selection.csv
└── src/                        # Modular, production-grade NLP codebase
    ├── __init__.py
    ├── dataset.py              # Cleaning, leakage-free splitting & loaders
    ├── features.py             # Normalization, vectorizers & linguistic features
    ├── models.py               # Classical estimators & Hybrid pipeline
    └── explainability.py       # Linear feature contribution engine (x_i * w_i)
```

---

## 4. Quick Start

### 4.1 Prerequisites & Installation
Ensure Python 3.10+ is installed:
```bash
pip install -r requirements.txt
```

### 4.2 Train Models
Train and serialize all 6 classifiers from scratch:
```bash
python train.py
```

### 4.3 Evaluate Benchmarks
Run the full comparative evaluation suite, generate all benchmark CSVs and figures:
```bash
python evaluate.py
```

### 4.4 Run Predictions (CLI)
Classify sentiment for any Haryanvi sentence with transparent explainability:
```bash
# Single sentence inference with proposed Hybrid model
python predict.py --text "yo kaam ghna badiya se"

# Compare all 6 models side-by-side
python predict.py --text "yo kaam badiya konya" --compare_all
```

### 4.5 Launch Interactive Dashboard
Run the multi-module Streamlit research dashboard:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 5. Academic Documentation

For complete mathematical formulations, literature citations, and detailed error analyses:
- **Academic Project Report**: [`report/Project_Report.md`](report/Project_Report.md)
- **Publication PDF**: [`report/Project_Report.pdf`](report/Project_Report.pdf)

---

## 6. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
