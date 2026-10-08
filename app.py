"""Standalone Streamlit emotion detector; no IBM Watson API required."""
import pandas as pd
import streamlit as st

MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"
EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness"]

st.set_page_config(page_title="AI Emotion Detector", page_icon="🧠", layout="centered")
st.title("🧠 AI-Based Emotion Detection")
st.write("Analyze English text with a pretrained local emotion classification model.")

@st.cache_resource(show_spinner="Loading the emotion model (first run downloads it)...")
def load_model():
    from transformers import pipeline
    return pipeline("text-classification", model=MODEL_ID, top_k=None, device=-1)

def analyze(text):
    classifier = load_model()
    # Model supports up to 512 tokens; truncation prevents errors for longer inputs.
    output = classifier(text, truncation=True, max_length=512)
    if output and isinstance(output[0], list):
        output = output[0]
    return {entry["label"].lower(): float(entry["score"]) for entry in output}

text = st.text_area("Text to analyze", placeholder="I am so happy about my achievement!", height=140)
if st.button("Analyze Emotion", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("Please enter some text before analyzing.")
    else:
        try:
            with st.spinner("Analyzing emotions..."):
                scores = analyze(text.strip())
            dominant = max(scores, key=scores.get)
            st.success(f"Dominant emotion: {dominant.title()}")
            st.metric("Dominant emotion confidence", f"{scores[dominant]:.1%}")
            df = pd.DataFrame([{"Emotion": name.title(), "Score": score} for name, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)])
            st.subheader("Emotion Scores")
            st.bar_chart(df.set_index("Emotion"))
            st.dataframe(df.style.format({"Score": "{:.4f}"}), hide_index=True, use_container_width=True)
            if any(label not in EMOTIONS for label in scores):
                st.info("This model also predicts neutral and surprise, in addition to the five original Watson emotions.")
        except Exception as exc:
            st.error("Model could not be loaded or prediction failed. Ensure the dependencies are installed and that the model can be downloaded on the first run.")
            st.code(f"{type(exc).__name__}: {exc}")

st.divider()
st.caption("Uses j-hartmann/emotion-english-distilroberta-base. Scores are model estimates, not clinical assessments. First launch requires access to Hugging Face; subsequent launches use the cached model.")
