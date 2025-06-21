# evaluate_model.py
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Emotion labels (must match the order used during training)
label_map = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
model_path = "emotion_model"

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

# Load test dataset
df = pd.read_csv("test_data.csv")

# Check if necessary columns exist
if "text" not in df.columns or "label" not in df.columns:
    raise ValueError("CSV must contain 'text' and 'label' columns.")

true_labels = []
pred_labels = []

# Predict for each example
for _, row in df.iterrows():
    text = row["text"]
    true_label = row["label"]

    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    pred_labels.append(pred)
    true_labels.append(label_map.index(true_label))  # Convert label string to index

# Accuracy
acc = accuracy_score(true_labels, pred_labels)
print(f"\n✅ Accuracy: {acc * 100:.2f}%")

# Classification report
print("\n📋 Classification Report:")
print(classification_report(true_labels, pred_labels, target_names=label_map))

# Confusion matrix
cm = confusion_matrix(true_labels, pred_labels)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_map, yticklabels=label_map)
plt.title("🧮 Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()
