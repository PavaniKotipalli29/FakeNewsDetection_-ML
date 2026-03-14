
import streamlit as st
import joblib

# ----- PAGE CONFIG -----
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# ----- CUSTOM CSS -----
st.markdown("""
<style>
.big-title {
    text-align:center;
    font-size:45px;
    font-weight:700;
    color:#2E86C1;
}
.subtitle {
    text-align:center;
    font-size:18px;
    color:gray;
    margin-bottom:30px;
}
.stTextArea textarea {
    border-radius:12px;
    border:2px solid #2E86C1;
}
.stButton>button {
    background-color:#2E86C1;
    color:white;
    border-radius:10px;
    height:3em;
    width:100%;
    font-size:18px;
}
.result-real {
    background-color:#D4EFDF;
    padding:15px;
    border-radius:10px;
    color:#1E8449;
    font-size:20px;
    text-align:center;
}
.result-fake {
    background-color:#FADBD8;
    padding:15px;
    border-radius:10px;
    color:#C0392B;
    font-size:20px;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

# ----- LOAD MODEL -----
vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("model.pkl")

# ----- UI -----
st.markdown('<div class="big-title">📰 Fake News Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI Powered News Authenticity Checker</div>', unsafe_allow_html=True)

inputn = st.text_area("✍ Enter News Article", height=200)

if st.button("🔍 Check News Authenticity"):
    if inputn.strip():

        transform_input = vectorizer.transform([inputn])
        prediction = model.predict(transform_input)

        if prediction[0] == 1:
            st.markdown('<div class="result-real">✅ This News Looks REAL</div>', unsafe_allow_html=True)
            st.balloons()

        else:
            st.markdown('<div class="result-fake">❌ This News Looks FAKE</div>', unsafe_allow_html=True)

    else:
        st.warning("⚠ Please enter some news text")

