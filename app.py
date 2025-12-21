
# sms_spam_app.py
import os
import string
import joblib
import streamlit as st

import nltk
from nltk.corpus import stopwords

# -------------------------------
# 0. Ensure NLTK stopwords are available
# -------------------------------
# Try to load stopwords; if missing, download them once.
try:
    _ = stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# -------------------------------
# 1. Load saved models & transformers (cached)
# -------------------------------
@st.cache_resource
def load_artifacts():
    try:
        path = "./models"
        spam_detect_model = joblib.load(f'{path}/spam_detect_model.pkl')
        count_vectorizer = joblib.load(f'{path}/count_vectorizer.pkl')
        tfidf_transformer = joblib.load(f'{path}/tfidf_transformer.pkl')
        ig_selector = joblib.load(f'{path}/ig_selector.pkl')
        return spam_detect_model, count_vectorizer, tfidf_transformer, ig_selector
    except Exception as e:
        st.error(f"Failed to load model artifacts: {e}")
        st.stop()

spam_detect_model, vectorizer, tfidf_transformer, selector = load_artifacts()

# -------------------------------
# 2. Preprocessing function
# -------------------------------
def text_preprocess(message: str) -> str:
    """
    Lowercase, remove punctuation, remove non-alpha tokens and English stopwords.
    Returns a cleaned string appropriate for the CountVectorizer.
    """
    if not isinstance(message, str):
        return ""
    nopunc = ''.join([c for c in message if c not in string.punctuation]).lower()
    return ' '.join([w for w in nopunc.split() if w.isalpha() and w not in stopwords.words('english')])

# -------------------------------
# 3. Prediction function
# -------------------------------
def predict_message(message: str):
    """
    Full pipeline: preprocess -> BoW -> TF-IDF -> selector -> model.predict
    Returns (label, proba_dict)
    """
    msg_processed = text_preprocess(message)

    # If everything got stripped (e.g., only punctuation/stopwords), fall back to original
    if not msg_processed.strip():
        msg_processed = message.lower()

    try:
        msg_bow = vectorizer.transform([msg_processed])
        msg_tfidf = tfidf_transformer.transform(msg_bow)
        msg_ig = selector.transform(msg_tfidf)
        prediction = spam_detect_model.predict(msg_ig)[0]

        # Try probabilities if available
        proba_dict = None
        if hasattr(spam_detect_model, "predict_proba"):
            probs = spam_detect_model.predict_proba(msg_ig)[0]
            # Assuming classes are like [0,1] or ['ham','spam']
            classes = list(spam_detect_model.classes_)
            proba_dict = {str(c): float(p) for c, p in zip(classes, probs)}
        return prediction, proba_dict
    except Exception as e:
        raise RuntimeError(f"Pipeline error: {e}")

# -------------------------------
# 4. Streamlit UI
# -------------------------------
st.set_page_config(page_title="SMS Spam Detection", page_icon="✉️", layout="centered")

st.title("✉️ SMS Spam Detection App")
st.write("Enter an SMS message below to predict whether it's **spam** or **ham**.")

with st.sidebar:
    st.header("Settings")
    show_probs = st.checkbox("Show prediction probabilities (if available)", value=False)
    st.markdown("---")
    st.caption("Model files are expected under `./joblib/`.")

default_text = "Congratulations! You've won a free ticket. Claim now!"
message = st.text_area("SMS message", value=default_text, height=150)

col1, col2 = st.columns([1,1])
with col1:
    if st.button("Predict"):
        if not message.strip():
            st.warning("Please enter a non-empty message.")
        else:
            try:
                label, proba_dict = predict_message(message)
                label_str = str(label)
                if label_str.lower() in ["spam", "1", "true"]:
                    st.error(f"Predicted label: **{label_str}**")
                else:
                    st.success(f"Predicted label: **{label_str}**")

                if show_probs and proba_dict is not None:
                    st.subheader("Probabilities")
                    st.write(proba_dict)
            except Exception as e:
                st.exception(e)

with col2:
    if st.button("Clear"):
        st.experimental_set_query_params()  # reset query params
        st.rerun()

st.markdown("---")
st.caption("Tip: If you deploy on a server, open the app via the URL printed in your terminal (`http://localhost:8501` by default).")
