import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ==================================================
# 1. CONFIG
# ==================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "Model"

MODEL_FILES = {
    "KNN": "best_knn.pkl",
    "SVC": "best_svm.pkl",
    "NB": "best_nb.pkl",
}

# name: (default value, type, min, max)  -> single source of truth
FEATURE_SPEC = {
    "message_length": (100, int, 0, None),
    "word_count": (20, int, 0, None),
    "num_urls": (0, int, 0, None),
    "num_digits": (0, int, 0, None),
    "num_special_chars": (2, int, 0, None),
    "spam_keyword_score": (0.0, float, 0.0, None),
    "legit_keyword_score": (0.0, float, 0.0, None),
    "sender_activity_score": (0.5, float, 0.0, None),
    "sender_account_age_days": (365, int, 0, None),
    "messages_sent_last_24h": (5, int, 0, None),
    "hour_of_day": (12, int, 0, 23),
    "day_of_week": (2, int, 0, 6),
}
FEATURES = list(FEATURE_SPEC.keys())

LABELS = {
    "message_length": "Message Length",
    "word_count": "Word Count",
    "num_urls": "Number of URLs",
    "num_digits": "Number of Digits",
    "num_special_chars": "Special Characters",
    "spam_keyword_score": "Spam Keyword Score",
    "legit_keyword_score": "Legitimate Keyword Score",
    "sender_activity_score": "Sender Activity Score",
    "sender_account_age_days": "Sender Account Age (days)",
    "messages_sent_last_24h": "Messages Sent Last 24 Hours",
    "hour_of_day": "Hour of Day (0-23)",
    "day_of_week": "Day of Week (0-6)",
}

# Heuristic word lists - replace with the ones used in your training pipeline
SPAM_WORDS = {"free", "win", "winner", "prize", "urgent", "click", "offer",
              "cash", "claim", "limited", "congratulations", "buy", "credit"}
LEGIT_WORDS = {"meeting", "thanks", "regards", "attached", "schedule",
               "project", "team", "report", "tomorrow", "please"}

st.set_page_config(page_title="Message Spam Intelligence",
                   page_icon="✉️", layout="centered")

# ==================================================
# VISUAL THEME
# ==================================================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f7f9fc 0%, #edf4f8 50%, #e8f0f7 100%);
        color: #172033;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        background: linear-gradient(135deg, #102a43 0%, #176b87 55%, #1f8a70 100%);
        padding: 2rem 2.2rem;
        border-radius: 20px;
        margin-bottom: 1.4rem;
        box-shadow: 0 12px 30px rgba(16, 42, 67, 0.18);
        color: white;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.25rem;
        font-weight: 750;
        letter-spacing: -0.5px;
    }

    .hero p {
        margin: 0.55rem 0 0;
        color: #e2f4f4;
        font-size: 1rem;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #102a43 0%, #163d5c 100%);
    }

    [data-testid="stSidebar"] * {
        color: #f4fbff !important;
    }

    div[data-testid="stForm"] {
        background: rgba(255,255,255,0.96);
        padding: 1.4rem 1.5rem 1.2rem;
        border: 1px solid #d8e4ec;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(16, 42, 67, 0.08);
    }

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div,
    textarea {
        border-radius: 10px !important;
        border-color: #c9d9e5 !important;
        background-color: #fbfdff !important;
    }

    div[data-baseweb="input"] > div:focus-within,
    div[data-baseweb="select"] > div:focus-within {
        border-color: #176b87 !important;
        box-shadow: 0 0 0 1px #176b87 !important;
    }

    button[kind="primary"] {
        border-radius: 10px;
        border: none;
        background: linear-gradient(90deg, #176b87, #1f8a70);
        color: white;
        font-weight: 700;
        min-height: 46px;
        box-shadow: 0 7px 16px rgba(23, 107, 135, 0.24);
    }

    button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 20px rgba(23, 107, 135, 0.30);
    }

    button[data-baseweb="tab"] {
        font-weight: 650;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #d8e4ec;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        box-shadow: 0 7px 20px rgba(16, 42, 67, 0.07);
    }

    div[data-testid="stMetricValue"] {
        color: #102a43;
        font-weight: 750;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    h2, h3 {
        color: #102a43 !important;
    }

    .section-card {
        background: white;
        border: 1px solid #d8e4ec;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0 1rem;
    }
</style>
""", unsafe_allow_html=True)



# ==================================================
# 2. MODEL LAYER
# ==================================================
@st.cache_resource
def load_models():
    loaded, problems = {}, []
    for name, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename
        if not path.is_file():
            problems.append(f"{name}: file not found ({path})")
            continue
        try:
            loaded[name] = joblib.load(path)
        except Exception as e:
            problems.append(f"{name}: could not load - {e}")
    return loaded, problems


# ==================================================
# 3. SERVICE LAYER
# ==================================================
def normalize_label(value):
    v = str(value).strip().lower()
    if v in {"1", "spam", "true", "yes"}:
        return "Spam"
    if v in {"0", "ham", "not spam", "false", "no"}:
        return "Ham"
    return None


def extract_features(text: str) -> dict:
    """Heuristic extraction of content-based features from raw text."""
    words = re.findall(r"[A-Za-z']+", text.lower())
    n_words = max(len(words), 1)
    return {
        "message_length": len(text),
        "word_count": len(text.split()),
        "num_urls": len(re.findall(r"(https?://|www\.)\S+", text, re.I)),
        "num_digits": sum(c.isdigit() for c in text),
        "num_special_chars": sum(not c.isalnum() and not c.isspace()
                                 for c in text),
        "spam_keyword_score": round(
            sum(w in SPAM_WORDS for w in words) / n_words, 4),
        "legit_keyword_score": round(
            sum(w in LEGIT_WORDS for w in words) / n_words, 4),
    }


def spam_class_index(model):
    """Index of the 'spam' class within model.classes_, if identifiable."""
    classes = getattr(model, "classes_", None)
    if classes is None:
        return None
    for i, c in enumerate(classes):
        if normalize_label(c) == "Spam":
            return i
    return None


def predict_frame(model, df: pd.DataFrame):
    """Return (labels, spam_confidence or None) for every row in df."""
    X = df[FEATURES]
    raw = model.predict(X)
    labels = [normalize_label(p) or str(p) for p in raw]

    conf = None
    idx = spam_class_index(model)
    try:
        if hasattr(model, "predict_proba") and idx is not None:
            conf = model.predict_proba(X)[:, idx]
        elif hasattr(model, "decision_function"):
            # Squash margin to 0-1 (not a calibrated probability)
            d = model.decision_function(X)
            if getattr(d, "ndim", 1) == 1:
                conf = 1 / (1 + pow(2.718281828, -d))
    except Exception:
        conf = None
    return labels, conf


def show_verdict(label: str):
    if label == "Spam":
        st.error("🚨 Spam Message")
    elif label == "Ham":
        st.success("✅ Ham (Not Spam)")
    else:
        st.info(f"Predicted class: {label}")


# ==================================================
# 4. SESSION STATE (defaults for form widgets)
# ==================================================
for feat, (default, *_rest) in FEATURE_SPEC.items():
    st.session_state.setdefault(feat, default)


def autofill_from_text():
    text = st.session_state.get("raw_text", "")
    for k, v in extract_features(text).items():
        typ = FEATURE_SPEC[k][1]
        st.session_state[k] = typ(v)


# ==================================================
# 5. HEADER + SIDEBAR
# ==================================================
models, problems = load_models()

st.markdown("""
<div class="hero">
    <h1>✉️ Message Spam Intelligence</h1>
    <p>Detect Spam or Ham using trained machine learning models.</p>
</div>
""", unsafe_allow_html=True)

for p in problems:
    st.warning(p)

if not models:
    st.error("No models could be loaded. Check the Model folder and "
             "retrain incompatible files in your current environment.")
    st.stop()

st.sidebar.header("Model Settings")
model_name = st.sidebar.selectbox("Choose Classifier", list(models.keys()))
compare_all = st.sidebar.checkbox("Compare all models", value=False)
st.sidebar.caption(f"Models available: {len(models)}")

tab_single, tab_batch = st.tabs(["Single Message", "Batch (CSV)"])

# ==================================================
# 6. TAB 1 - SINGLE MESSAGE
# ==================================================
with tab_single:
    with st.expander("Auto-fill content features from message text"):
        st.text_area("Paste a message", key="raw_text", height=100)
        st.button("Extract features", on_click=autofill_from_text)
        st.caption("Fills the text-based features only. Sender and time "
                   "features stay as entered below. Keyword scores use "
                   "simple heuristics and may differ from your training data.")

    st.subheader("📝 Message Features")
    with st.form("spam_form"):
        cols = st.columns(2)
        values = {}
        for i, feat in enumerate(FEATURES):
            default, typ, lo, hi = FEATURE_SPEC[feat]
            kwargs = {"min_value": lo, "key": feat}
            if hi is not None:
                kwargs["max_value"] = hi
            if typ is float:
                kwargs["step"] = 0.1
            with cols[i % 2]:
                values[feat] = st.number_input(LABELS[feat], **kwargs)
        submitted = st.form_submit_button("Analyze Message",
                                          use_container_width=True)

    if submitted:
        df = pd.DataFrame([values], columns=FEATURES)
        st.subheader("📊 Prediction Result")

        targets = list(models.keys()) if compare_all else [model_name]
        results = []
        for name in targets:
            try:
                labels, conf = predict_frame(models[name], df)
                results.append({
                    "Model": name,
                    "Prediction": labels[0],
                    "Spam confidence": (f"{conf[0]:.1%}"
                                        if conf is not None else "n/a"),
                })
            except Exception as e:
                st.error(f"{name} failed: check the 12 features, their "
                         "order and preprocessing.")
                st.exception(e)

        if results:
            if compare_all:
                votes = [r["Prediction"] for r in results]
                majority = max(set(votes), key=votes.count)
                st.markdown("**Majority vote**")
                show_verdict(majority)
                st.dataframe(pd.DataFrame(results), hide_index=True,
                             use_container_width=True)
            else:
                r = results[0]
                show_verdict(r["Prediction"])
                st.caption(f"Model: {r['Model']} | "
                           f"Spam confidence: {r['Spam confidence']}")

# ==================================================
# 7. TAB 2 - BATCH
# ==================================================
with tab_batch:
    st.subheader("📦 Batch Prediction")
    st.caption("Upload a CSV containing these columns: " + ", ".join(FEATURES))
    file = st.file_uploader("CSV file", type="csv")

    if file is not None:
        try:
            data = pd.read_csv(file)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        missing = [c for c in FEATURES if c not in data.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            labels, conf = predict_frame(models[model_name], data)
            out = data.copy()
            out["prediction"] = labels
            if conf is not None:
                out["spam_confidence"] = conf.round(4)

            c1, c2 = st.columns(2)
            c1.metric("Rows scored", len(out))
            c2.metric("Spam rate", f"{(out['prediction'] == 'Spam').mean():.1%}")

            st.dataframe(out, use_container_width=True)
            st.download_button("Download results",
                               out.to_csv(index=False).encode("utf-8"),
                               "spam_predictions.csv", "text/csv")