import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Load model and tokenizer
MODEL_PATH = "emotion_model"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# Emotion labels and emojis
label_map = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
emoji_map = {
    'anger': '😠', 'fear': '😨', 'joy': '😊',
    'love': '❤️', 'sadness': '😢', 'surprise': '😲'
}

# Page configuration
st.set_page_config(page_title="Emotion Detection", layout="centered")

# Title and description
st.title("💬😊 Emotion Detection from Text using BERT")
st.markdown("Enter any sentence below and the model will detect the emotion it conveys using a fine-tuned BERT model.")

# Input section
st.subheader("📝 Input Text")
user_input = st.text_area("Type your sentence here:", height=150)

# Predict emotion
if st.button("🔍 Predict Emotion"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        # Tokenize input
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=128)

        # Get model predictions
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1).squeeze()

        pred_idx = torch.argmax(probs).item()
        pred_emotion = label_map[pred_idx]
        emoji = emoji_map[pred_emotion]

        # Show prediction
        st.success(f"**Predicted Emotion: {pred_emotion.upper()} {emoji}**")

        # Confidence scores
        st.subheader("📈 Confidence Scores")
        for i, score in enumerate(probs):
            emotion = label_map[i]
            emotion_emoji = emoji_map[emotion]
            st.write(f"{emotion.capitalize()} {emotion_emoji}: **{score.item():.2f}**")

        # Bar chart
        st.subheader("📊 Emotion Probabilities")
        fig, ax = plt.subplots()
        emotion_labels_with_emojis = [f"{label.capitalize()} {emoji_map[label]}" for label in label_map]
        ax.bar(emotion_labels_with_emojis, probs.numpy(), color="skyblue")
        ax.set_ylim([0, 1])
        ax.set_ylabel("Probability")
        ax.set_title("Predicted Emotion Distribution")
        plt.xticks(rotation=45)
        st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using [BERT](https://huggingface.co/bert-base-uncased) and [Streamlit](https://streamlit.io)")
