"""
BoliSense: Dialect-Aware Sentiment Analysis for Haryanvi Text
Streamlit Research Application.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from src.explainability import BoliSenseInferenceEngine
from src.features import DialectNormalizer, word_tokenize, char_ngrams
from src.dataset import PROCESSED_DATA_DIR, CHALLENGE_DATA_PATH

st.set_page_config(
    page_title="BoliSense | Dialect Sentiment Analysis",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 1.9rem; font-weight: 700; color: #1a365d; margin-bottom: 0.2rem; }
    .sub-header { font-size: 0.95rem; color: #4a5568; margin-bottom: 1.2rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_engine():
    return BoliSenseInferenceEngine()


@st.cache_data
def load_data():
    results_dir = Path("results")
    from src.dataset import load_processed_splits
    train_df, val_df, test_df = load_processed_splits()
    corpus_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    data = {
        "corpus": corpus_df,
        "comparison": pd.read_csv(results_dir / "model_comparison.csv") if (results_dir / "model_comparison.csv").exists() else pd.DataFrame(),
        "category": pd.read_csv(results_dir / "category_results.csv") if (results_dir / "category_results.csv").exists() else pd.DataFrame(),
        "ablation": pd.read_csv(results_dir / "ablation_results.csv") if (results_dir / "ablation_results.csv").exists() else pd.DataFrame(),
        "cv": pd.read_csv(results_dir / "cross_validation.csv") if (results_dir / "cross_validation.csv").exists() else pd.DataFrame(),
        "challenge": pd.read_csv(results_dir / "challenge_results.csv") if (results_dir / "challenge_results.csv").exists() else pd.DataFrame(),
        "pairs": pd.read_csv(results_dir / "minimal_pair_results.csv") if (results_dir / "minimal_pair_results.csv").exists() else pd.DataFrame(),
    }
    return data


engine = get_engine()
app_data = load_data()

st.sidebar.markdown("## **BoliSense**")
st.sidebar.markdown("*Dialect-Aware Sentiment Analysis for Haryanvi Text*")
st.sidebar.markdown("---")

section = st.sidebar.radio(
    "Navigation",
    [
        "Sentiment Analysis",
        "Dataset Explorer",
        "Model Comparison",
        "Linguistic Analysis",
        "Results & Robustness"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Final Submission Ready • 100% Offline Classical NLP")


# 1. Sentiment Analysis
if section == "Sentiment Analysis":
    st.markdown('<div class="main-header">Haryanvi Sentiment Analysis & Explainability</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Enter Roman-script Haryanvi text to inspect sentiment prediction, local feature contributions, and linguistic cues.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col2:
        samples = [
            "Custom Input",
            "yo kaam ghna badiya se (Positive slang)",
            "yo kaam badiya konya (Negated positive -> Negative)",
            "yo kaam bekaar konya (Negated negative -> Positive)",
            "yo app slow se par kaam sahi kare se (Contrastive -> Positive)",
            "yo plan sahi se par execution weak se (Contrastive -> Negative)",
            "yo phone thikk se mast chal reya se (Spelling variant -> Positive)",
            "bhai bilkul bakwaas bana diya (Direct negative)"
        ]
        sel_sample = st.selectbox("Benchmark Examples:", samples)

    with col1:
        default_val = "yo kaam ghna badiya se" if sel_sample == "Custom Input" else sel_sample.split(" (")[0]
        text_input = st.text_area("Haryanvi Sentence:", value=default_val, height=90)
        sel_model = st.selectbox("Select Model Architecture:", list(engine.models.keys()))
        btn = st.button("Classify Sentiment", type="primary")

    if text_input.strip():
        res = engine.predict(text_input, model_name=sel_model)
        st.markdown("---")

        r1, r2, r3, r4 = st.columns(4)
        sentiment_color = "#22543d" if res["sentiment"] == "Positive" else "#742a2a"
        bg_color = "#c6f6d5" if res["sentiment"] == "Positive" else "#fed7d7"

        with r1:
            st.markdown(f"""
            <div style="background-color: {bg_color}; padding: 10px; border-radius: 6px; text-align: center;">
                <div style="font-size: 0.8rem; color: {sentiment_color}; font-weight: 500;">Prediction</div>
                <div style="font-size: 1.5rem; color: {sentiment_color}; font-weight: 700;">{res["sentiment"]}</div>
            </div>
            """, unsafe_allow_html=True)
        with r2:
            st.metric("Model Confidence", f"{res['confidence']*100:.1f}%")
        with r3:
            st.metric("Model Used", res["model"])
        with r4:
            st.metric("Word Count", str(len(text_input.split())))

        # Linear feature contributions
        contribs = res["explanation"]["contributions"]
        if contribs["top_positive"] or contribs["top_negative"]:
            st.markdown("#### Feature Contributions $(x_i \\times w_i)$")
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                st.markdown("**Positive Drivers (+):**")
                if contribs["top_positive"]:
                    st.dataframe(pd.DataFrame(contribs["top_positive"]), hide_index=True)
            with fc2:
                st.markdown("**Negative Drivers (-):**")
                if contribs["top_negative"]:
                    st.dataframe(pd.DataFrame(contribs["top_negative"]), hide_index=True)
            with fc3:
                st.markdown("**Decision Logit:**")
                st.write(f"- Net Logit ($z$): `{contribs['net_logit']}`")
                st.write(f"- Normalized Text: *'{res['normalized']}'*")

        # Linguistic Cues
        st.markdown("#### Linguistic Rule Detections")
        cues = res["explanation"]["cues"]
        lc1, lc2, lc3 = st.columns(3)
        with lc1:
            neg = cues["negation"]
            if neg["has_negation"]:
                st.warning(f"Negation Marker: `{neg['negation_tokens']}`")
            else:
                st.info("No negation markers found.")
        with lc2:
            con = cues["contrast"]
            if con["has_contrast"]:
                st.warning(f"Contrast Conjunction: `{con['contrast_marker']}`")
                st.caption(f"Resolving: *'{con['clause_2']}'*")
            else:
                st.info("No contrast markers found.")
        with lc3:
            lex = cues["lexicon"]
            st.write(f"- Pos Lexicon: `{lex['positive_hits'] if lex['positive_hits'] else 'None'}`")
            st.write(f"- Neg Lexicon: `{lex['negative_hits'] if lex['negative_hits'] else 'None'}`")

        # Multi-model comparison
        st.markdown("#### Multi-Model Side-by-Side Comparison")
        st.dataframe(pd.DataFrame(engine.predict_all(text_input)), hide_index=True)


# 2. Dataset Explorer
elif section == "Dataset Explorer":
    st.markdown('<div class="main-header">Corpus Explorer & Statistics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Overview and exploration of the curated 546-sentence Roman-script Haryanvi dataset.</div>', unsafe_allow_html=True)

    df = app_data["corpus"]
    if not df.empty:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Records", len(df))
        m2.metric("Negative Sentences", int((df["label"] == 0).sum()))
        m3.metric("Positive Sentences", int((df["label"] == 1).sum()))
        m4.metric("Linguistic Categories", len(df["type"].unique()))

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Sentiment Distribution")
            counts = df["label"].map({0: "Negative (0)", 1: "Positive (1)"}).value_counts()
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette=["#d95f02", "#2b5c8f"], legend=False, ax=ax)
            ax.set_ylabel("Sentences")
            st.pyplot(fig)
        with c2:
            st.markdown("##### Linguistic Category Distribution")
            t_counts = df["type"].value_counts()
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.barplot(x=t_counts.index, y=t_counts.values, hue=t_counts.index, palette="crest", legend=False, ax=ax)
            ax.set_ylabel("Sentences")
            st.pyplot(fig)

        st.markdown("##### Searchable Corpus")
        sel_cat = st.multiselect("Category Filter:", df["type"].unique(), default=df["type"].unique())
        st.dataframe(df[df["type"].isin(sel_cat)], height=260)


# 3. Model Comparison
elif section == "Model Comparison":
    st.markdown('<div class="main-header">Empirical Model Benchmarking</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Independent test set results comparing classical baselines, linear classifiers, and the proposed hybrid model.</div>', unsafe_allow_html=True)

    comp_df = app_data["comparison"]
    if not comp_df.empty:
        st.markdown("##### Primary Benchmark Table (Held-Out Test Set N=82)")
        st.dataframe(comp_df, hide_index=True)

        b1, b2 = st.columns([3, 2])
        with b1:
            if Path("results/figures/model_comparison.png").exists():
                st.image("results/figures/model_comparison.png", caption="Accuracy and Macro F1 Comparison")
        with b2:
            if Path("results/figures/confusion_matrix.png").exists():
                st.image("results/figures/confusion_matrix.png", caption="Proposed Hybrid Confusion Matrix")

    if not app_data["category"].empty:
        st.markdown("##### Category-Wise Accuracy (Proposed Hybrid)")
        st.dataframe(app_data["category"], hide_index=True)


# 4. Linguistic Analysis
elif section == "Linguistic Analysis":
    st.markdown('<div class="main-header">NLP Pipeline & Linguistic Processing</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Live demonstration of tokenization strategies, dialect normalization, and feature extraction.</div>', unsafe_allow_html=True)

    demo_text = st.text_input("Enter sentence to dissect:", value="yo kaam ghnaa badhiya se par thikk se chalya konya")
    if demo_text:
        normalizer = DialectNormalizer()
        norm_text = normalizer.normalize(demo_text)
        replaced = normalizer.get_replaced_tokens(demo_text)

        t1, t2 = st.columns(2)
        with t1:
            st.markdown("##### Dialect Normalization")
            st.write(f"- Original: `{demo_text}`")
            st.write(f"- Normalized: `{norm_text}`")
            if replaced:
                st.success(f"Replaced tokens: {replaced}")
            else:
                st.info("No normalization replacements required.")
        with t2:
            st.markdown("##### Multi-Level Tokenization")
            words = word_tokenize(demo_text)
            c_grams = char_ngrams(demo_text, (3, 4))
            st.write(f"- Word Tokens ({len(words)}): `{words}`")
            st.write(f"- Character N-Grams sample ({len(c_grams)} total): `{c_grams[:6]}`...")


# 5. Results & Robustness
elif section == "Results & Robustness":
    st.markdown('<div class="main-header">Robustness Evaluation & Ablation Study</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluation on the frozen 65-sample Linguistic Challenge Dataset (Minimal Pairs) and component ablations.</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Challenge Dataset & Minimal Pairs", "Ablation Study", "Grouped Cross-Validation"])

    with tab1:
        st.markdown("##### Linguistic Challenge Benchmark (N=65)")
        st.dataframe(app_data["challenge"], hide_index=True)
        st.markdown("##### Minimal-Pair Diagnostics")
        st.dataframe(app_data["pairs"], hide_index=True)

    with tab2:
        st.markdown("##### 8-Way Additive Component Ablation")
        st.dataframe(app_data["ablation"], hide_index=True)

    with tab3:
        st.markdown("##### Grouped-by-Template 5-Fold Cross-Validation")
        st.dataframe(app_data["cv"], hide_index=True)
