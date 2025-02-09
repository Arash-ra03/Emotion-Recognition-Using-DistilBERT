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
    under_threshold = (probs < best_thr)
    distances = best_thr - probs

    if any(under_threshold):
        closest_idx = np.argmin(distances[under_threshold])
        actual_idx = np.arange(len(labels))[under_threshold][closest_idx]
        closest_emotion = labels[actual_idx]
        closest_distance = distances[actual_idx]
    else:
        closest_emotion, closest_distance = None, None

    return predicted_labels, probs, closest_emotion, closest_distance


st.title("Emotion Classification")
st.write("Enter a sentence, and the model will predict its emotional tone.")

user_input = st.text_area("Enter text here:", "")

if st.button("Predict Emotion"):
    if user_input.strip():
        pred_labels, probs, closest_emotion, closest_distance = predict_single_text(user_input)


        st.subheader("Predicted Emotions:")
        st.write(pred_labels if pred_labels else "No strong emotion detected.")


        if closest_emotion:
            st.subheader("Emotion Closest to Being Predicted (Under Threshold):")
            st.write(f"**{closest_emotion}** (Distance: {closest_distance:.4f})")
        else:
            st.write("All emotions exceeded their thresholds.")


        st.subheader("Probabilities:")
        for label, prob, thr in zip(labels, probs, best_thr):
            st.write(f"{label}: {prob:.4f} (Threshold: {thr:.4f})")

    else:
        st.warning("Please enter some text.")
