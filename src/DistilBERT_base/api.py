import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify
from transformers import DistilBertTokenizer
from src.DistilBERT_base.model import DistilBERTBaseUncased
from src.DistilBERT_base import config

# ==== 1. Model and Tokenizer Setup ==== #
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = config.MODEL_SAVE_PATH  
MODEL = DistilBERTBaseUncased()
MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
MODEL.to(DEVICE)
MODEL.eval()
TOKENIZER = DistilBertTokenizer.from_pretrained(config.BERT_PATH)


# ==== 2. PREDICTION FUNCTION ==== #
def predict_sentiment(text):
    # Tokenize the input text
    inputs = TOKENIZER.encode_plus(
        text,
        None,
        add_special_tokens=True,
        max_length=config.MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt', # Return PyTorch tensors
        return_attention_mask=True
    )
    
    # Move inputs to CPU/GPU
    ids = inputs["input_ids"].to(DEVICE)
    mask = inputs["attention_mask"].to(DEVICE)
    
    with torch.no_grad():
        # Get raw output
        outputs = MODEL(input_ids=ids, attention_mask=mask)
        # Apply sigmoid to convert to probability (since it's binary classification)
        probs = torch.sigmoid(outputs).cpu().item() 
        
        # Convert probability to final sentiment label
        sentiment = "Positive" if probs >= 0.5 else "Negative"
        
        return sentiment, probs

# ==== 3. FLASK APP SETUP ==== #
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Expects JSON data: {"text": "This movie was absolutely amazing."}
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Invalid request: 'text' field is required."}), 400

    text = data['text']
    sentiment, probability = predict_sentiment(text)

    return jsonify({
        "input_text": text,
        "sentiment": sentiment,
        "probability": round(probability, 4)
    })

if __name__ == "__main__":
    print("--- Loading Model... ---")
    # The model is loaded once at the start (above)
    print("--- Starting Flask App on http://127.0.0.1:5000 ---")
    app.run(host='0.0.0.0', port=5000, debug=False)

# Run the Flask APP `python api.py`
# Test the endpoint with another terminal:
# curl -X POST -H "Content-Type: application/json" -d "{\"text\": \"This is the best experience I've ever had.\"}" http://127.0.0.1:5000/predict