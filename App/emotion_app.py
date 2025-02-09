import streamlit as st
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn as nn


# Paths to saved models/thresholds
distilbert_model_path = "final_emotion_model"
mlp_weights_path = "final_mlp.pth"
thresholds_path = "final_thresholds.npy"

# Load DistilBERT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
distilbert_model = AutoModelForSequenceClassification.from_pretrained(distilbert_model_path)
distilbert_model.to(device)
distilbert_model.eval()

tokenizer = AutoTokenizer.from_pretrained(distilbert_model_path)

# Define MLP model
def build_mlp(input_dim, hidden_dim, num_labels, dropout, n_layers):
    if n_layers == 1:
        return nn.Sequential(nn.Linear(input_dim, num_labels))
    elif n_layers == 2:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels)
        )
    else:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels)
        )

# Define model parameters
hidden_dim = 256
dropout = 0.2
n_layers = 1
num_labels = 10

mlp_model = build_mlp(768, hidden_dim, num_labels, dropout, n_layers).to(device)
mlp_model.load_state_dict(torch.load(mlp_weights_path, map_location=device))
mlp_model.eval()

# Load thresholds
best_thr = np.load(thresholds_path)

# Labels list
labels = [
    "praise", "amusement", "anger", "disapproval",
    "confusion", "interest", "sadness", "fear", "joy", "love"
]

# Prediction function
def predict_single_text(text):
    encoding = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = distilbert_model.distilbert(input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]

    with torch.no_grad():
        logits = mlp_model(cls_emb)

    probs = torch.sigmoid(logits).cpu().numpy().flatten()
    predictions = (probs >= best_thr).astype(int)

    # Get predicted labels
    predicted_labels = [labels[i] for i, p in enumerate(predictions) if p == 1]

    # Find the emotion under its threshold that is closest to it
    under_threshold = (probs < best_thr)  # Boolean mask for emotions below threshold
    distances = best_thr - probs  # Compute distance from threshold

    if any(under_threshold):  # Ensure there's at least one emotion under threshold
        closest_idx = np.argmin(distances[under_threshold])  # Index in filtered list
        actual_idx = np.arange(len(labels))[under_threshold][closest_idx]  # Get real index
        closest_emotion = labels[actual_idx]
        closest_distance = distances[actual_idx]
    else:
        closest_emotion, closest_distance = None, None  # If all are above threshold

    return predicted_labels, probs, closest_emotion, closest_distance
# Custom CSS for styling
st.markdown("""
    <style>
        .main {background-color: #f0f2f6;}
        .stTextArea textarea {border: 2px solid #4a90e2;}
        .emotion-box {padding: 1rem; border-radius: 10px; margin: 0.5rem 0;}
        .threshold-info {color: #666; font-size: 0.9rem;}
        .highlight {background-color: #fff2ac; padding: 0.2rem 0.5rem; border-radius: 5px;}
    </style>
""", unsafe_allow_html=True)

# Emoji mapping for emotions
EMOJI_MAP = {
    "praise": "👏",
    "amusement": "😄",
    "anger": "😠",
    "disapproval": "👎",
    "confusion": "😕",
    "interest": "🧐",
    "sadness": "😢",
    "fear": "😨",
    "joy": "😊",
    "love": "❤️"
}

# Color mapping for emotion boxes
COLOR_MAP = {
    "praise": "#e3f2fd",
    "amusement": "#f0f4c3",
    "anger": "#ffcdd2",
    "disapproval": "#eceff1",
    "confusion": "#d1c4e9",
    "interest": "#c8e6c9",
    "sadness": "#bbdefb",
    "fear": "#ffccbc",
    "joy": "#fff9c4",
    "love": "#f8bbd0"
}

# ... [Keep the existing model loading code the same] ...

# Streamlit UI
st.title("🎭 Emotion Detection Dashboard")
st.markdown("""
    *"Words have emotions, let's uncover them!"*  
    Enter text below to analyze its emotional content.
""")

user_input = st.text_area("📝 **Enter your text here:**", "", height=150)

if st.button("🔍 Analyze Emotions", help="Click to analyze the emotional content"):
    if user_input.strip():
        pred_labels, probs, closest_emotion, closest_distance = predict_single_text(user_input)

        # Display predicted emotions with emojis
        st.subheader("✨ Detected Emotions")
        if pred_labels:
            cols = st.columns(3)
            for idx, emotion in enumerate(pred_labels):
                with cols[idx % 3]:
                    st.markdown(
                        f"<div class='emotion-box' style='background-color: {COLOR_MAP[emotion]}'>"
                        f"<h3>{EMOJI_MAP[emotion]} {emotion.capitalize()}</h3>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
        else:
            st.markdown("<div class='highlight'>🤷 No strong emotions detected</div>", unsafe_allow_html=True)

        # Visual progress bars for probabilities
        st.subheader("📊 Emotion Probabilities")
        for label, prob, thr in zip(labels, probs, best_thr):
            col1, col2 = st.columns([2, 5])
            with col1:
                st.markdown(f"{EMOJI_MAP[label]} **{label.capitalize()}**")
            with col2:
                bar = st.progress(0)
                bar.progress(int(prob * 100),
                             text=f"{prob:.2f} (Threshold: {thr:.2f}) {'✅' if prob >= thr else '❌'}")

        # Closest under-threshold emotion
        if closest_emotion:
            st.subheader("🔍 Almost There...")
            st.markdown(
                f"<div style='background-color: #fff3e0; padding: 1rem; border-radius: 10px;'>"
                f"💡 Closest undetected emotion: <strong>{EMOJI_MAP[closest_emotion]} {closest_emotion.capitalize()}</strong><br>"
                f"Distance from threshold: {abs(closest_distance):.4f}"
                "</div>",
                unsafe_allow_html=True
            )

        # Technical details expander
        with st.expander("📚 Technical Details"):
            st.markdown("""
                **Model Architecture:**
                - DistilBERT base model
                - Custom MLP classifier
                - Per-label optimized thresholds
            """)

    else:
        st.warning("⚠️ Please enter some text to analyze!")

# Add footer
st.markdown("---",unsafe_allow_html=True)
