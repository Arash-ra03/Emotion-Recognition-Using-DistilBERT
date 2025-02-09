# single_inference.py (example script)
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn as nn

# 1) Paths to saved models/thresholds
distilbert_model_path = "final_emotion_model"  # Your DistilBERT
mlp_weights_path      = "final_mlp.pth"       # Saved MLP state_dict
thresholds_path       = "final_thresholds.npy" # Saved thresholds array

# 2) Load DistilBERT
distilbert_model = AutoModelForSequenceClassification.from_pretrained(distilbert_model_path)
distilbert_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
distilbert_model.to(device)

tokenizer = AutoTokenizer.from_pretrained(distilbert_model_path)

# 3) Define your MLP architecture EXACTLY as you did before
def build_mlp(input_dim, hidden_dim, num_labels, dropout, n_layers):
    if n_layers == 1:
        return nn.Sequential(
            nn.Linear(input_dim, num_labels)
        )
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

# 4) Build an MLP with the same hyperparameters you ultimately used
#    (check "best_params" from your training or final loaded config)
#    Suppose you ended up with hidden_dim=256, dropout=0.2, n_layers=2, etc.
hidden_dim = 256      # Example
dropout = 0.2         # Example
n_layers = 1          # Example
num_labels = 10       # Must match the actual number of labels you trained on
mlp_model = build_mlp(768, hidden_dim, num_labels, dropout, n_layers).to(device)

# 5) Load the MLP state_dict
mlp_model.load_state_dict(torch.load(mlp_weights_path, map_location=device))
mlp_model.eval()

# 6) Load the thresholds
best_thr = np.load(thresholds_path)
print("Loaded thresholds:", best_thr)

# 7) Provide the label list in the same order as training
labels = [
    "praise", "amusement", "anger", "disapproval",
    "confusion", "interest", "sadness", "fear", "joy", "love"
]

# 8) Define an inference function
def predict_single_text(text: str):
    """
    1) Tokenize text
    2) Extract DistilBERT hidden state for [CLS]-like embedding
    3) Feed embedding to MLP
    4) Apply sigmoid + best_thr => multi-label predictions
    5) Return predicted label names
    """
    # Tokenize
    encoding = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # DistilBERT hidden states (ignore classification head)
    with torch.no_grad():
        outputs = distilbert_model.distilbert(input_ids, attention_mask=attention_mask)
        # shape => (batch_size=1, seq_len, hidden_dim=768)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # (1, 768)

    # MLP forward
    with torch.no_grad():
        logits = mlp_model(cls_emb)  # shape => (1, num_labels)

    # Sigmoid + threshold
    probs = torch.sigmoid(logits).cpu().numpy().flatten()
    predictions = (probs >= best_thr).astype(int)

    # Map to label names
    predicted_labels = [labels[i] for i, p in enumerate(predictions) if p == 1]
    return predicted_labels, probs

# 9) Try it out
text = "Fuck I’m hungry now."
pred_labels, probs = predict_single_text(text)
print("Text:", text)
print("Predicted labels:", pred_labels)
print("Probabilities:", probs)

