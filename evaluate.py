import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, classification_report
import torch

def main():
    # Load test data
    test_df = pd.read_csv("test.csv")
    test_dataset = Dataset.from_pandas(test_df)
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./model_final")
    model = AutoModelForSequenceClassification.from_pretrained("./model_final")
    model.eval()
    
    # Tokenize
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)
    
    test_dataset = test_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.rename_column("label", "labels")
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    # Run predictions
    all_preds = []
    all_labels = []
    
    for i in range(len(test_dataset)):
        item = test_dataset[i]
        input_ids = item["input_ids"].unsqueeze(0)
        attention_mask = item["attention_mask"].unsqueeze(0)
        labels = item["labels"].item()
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = torch.argmax(outputs.logits, dim=1).item()
        
        all_preds.append(pred)
        all_labels.append(labels)
    
    # Print results
    print("=" * 60)
    print("FINAL EVALUATION RESULTS ON TEST SET")
    print("=" * 60)
    
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary")
    acc = accuracy_score(all_labels, all_preds)
    
    print(f"\nAccuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    print(f"\nTrue Negatives (Correct Ham): {cm[0,0]}")
    print(f"False Positives (Ham as Phishing): {cm[0,1]}")
    print(f"False Negatives (Phishing as Ham): {cm[1,0]}")
    print(f"True Positives (Correct Phishing): {cm[1,1]}")
    
    print("\n" + "=" * 60)
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Ham", "Phishing"]))

if __name__ == "__main__":
    main()
