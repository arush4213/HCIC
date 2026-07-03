import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def predict(text):
    tokenizer = AutoTokenizer.from_pretrained("./model_final")
    model = AutoModelForSequenceClassification.from_pretrained("./model_final")
    model.eval()
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.softmax(outputs.logits, dim=1)
    label = torch.argmax(probs).item()
    confidence = probs[0][label].item()
    
    result = "🚨 PHISHING" if label == 1 else "✅ HAM (Legitimate)"
    
    print(f"\nSMS: {text}")
    print(f"Classification: {result}")
    print(f"Confidence: {confidence*100:.2f}%")
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hinglish Phishing SMS Classifier")
    parser.add_argument("--sms", type=str, required=True, help="SMS text to classify")
    
    args = parser.parse_args()
    predict(args.sms)
