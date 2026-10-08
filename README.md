# AI Emotion Detector — Streamlit (local inference)

Replaces the unreachable IBM Watson course endpoint with the pretrained Hugging Face model `j-hartmann/emotion-english-distilroberta-base`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
python -m pip install -r requirements.txt
streamlit run app.py
```

On Windows, activate with `.venv\\Scripts\\activate`.

The model downloads on first use (internet needed); afterwards it is cached locally. The model predicts anger, disgust, fear, joy, sadness, neutral and surprise. These scores are not directly comparable to the original Watson model's scores.
