# Automated Hinglish BERT Classifier for Vishing and Smishing Phishing Campaigns

A multilingual DistilBERT-based classifier that detects **Hinglish (Hindi written in Latin script)** phishing SMS messages with high accuracy. The model is fine-tuned to distinguish phishing (smishing) messages from legitimate SMS by learning code-switched linguistic patterns commonly found in Indian cyber fraud campaigns.

---

# Overview

Traditional spam filters are primarily trained on English text and often fail to detect phishing messages written in **Hinglish**, where Hindi is expressed using the Latin alphabet.

This project fine-tunes **`distilbert-base-multilingual-cased`** on a curated SMS dataset to perform binary classification between phishing and legitimate messages.

The project aims to support defensive cybersecurity applications such as:

- SMS filtering
- Mobile security applications
- Telecom fraud detection
- Cybersecurity research
- Awareness and educational tools

---

# Key Results

| Metric | Score |
|---------|--------|
| Accuracy | **99.67%** |
| Precision | **100.00%** |
| Recall | **99.31%** |
| F1 Score | **99.65%** |

Tested on **302 unseen SMS messages**.

---

# Features

- Hinglish phishing detection
- Multilingual DistilBERT backbone
- Binary SMS classification
- Command-line prediction interface
- Evaluation pipeline
- Fine-tuning using HuggingFace Transformers
- PyTorch implementation

---

# Dataset

## Statistics

| Category | Count |
|----------|------:|
| Total Samples | 2,098 |
| Phishing (Label 1) | 970 |
| Legitimate / Ham (Label 0) | 1,128 |

Dataset Split:

- Training: 70%
- Validation: 15%
- Testing: 15%

### Dataset Includes

- Hinglish phishing SMS
- KYC scam messages
- Fake electricity bill alerts
- UPI suspension warnings
- PAN/Aadhaar scams
- Fake lottery messages
- Banking alerts
- OTP messages
- Delivery notifications
- Recharge confirmations
- Code-switched Hindi-English SMS

---

# Installation

## Requirements

- Python 3.8+
- 4GB RAM minimum
- GPU (optional but recommended)

## Clone Repository

```bash
git clone https://github.com/yourusername/hinglish-bert-classifier.git

cd hinglish-bert-classifier
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Usage

## Train the Model

```bash
python train.py
```

Training pipeline:

- Load dataset
- Tokenize SMS messages
- Split dataset
- Fine-tune DistilBERT
- Evaluate validation performance
- Save trained checkpoint

Expected training time:

- GPU: ~10 minutes
- CPU: ~30 minutes

---

## Evaluate

```bash
python evaluate.py
```

Outputs:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

---

## Predict

```bash
python predict.py --sms "Aapka KYC block ho gaya hai turant link pe click karein"
```

Example output:

```
SMS:
Aapka KYC block ho gaya hai turant link pe click karein

Classification:
🚨 PHISHING

Confidence:
99.97%
```

---

# Example Predictions

## Phishing

```bash
python predict.py --sms "Congratulations aapne 50000 rupaye jeete hain abhi claim karein"
```

Output:

```
🚨 PHISHING
Confidence: 99.97%
```

---

## Legitimate

```bash
python predict.py --sms "Your OTP for SBI transaction is 482910 do not share"
```

Output:

```
✅ HAM (Legitimate)

Confidence: 98.56%
```

---

# Technical Architecture

## Base Model

| Property | Value |
|----------|-------|
| Model | distilbert-base-multilingual-cased |
| Framework | HuggingFace Transformers |
| Backend | PyTorch |
| Parameters | ~67 Million |
| Supported Languages | 104 |
| Max Token Length | 128 |

---

## Training Configuration

| Hyperparameter | Value |
|---------------|------|
| Batch Size | 16 |
| Epochs | 4 |
| Learning Rate | 5e-5 |
| Optimizer | Adam |

---

# Performance

## Test Results

| Metric | Value |
|---------|-------|
| Accuracy | 99.67% |
| Correct Predictions | 301 / 302 |

### Confusion Matrix

| | Predicted Legitimate | Predicted Phishing |
|----------------------|----------------:|----------------:|
| Actual Legitimate | 157 | 0 |
| Actual Phishing | 1 | 144 |

This corresponds to:

- True Negatives: **157**
- True Positives: **144**
- False Positives: **0**
- False Negatives: **1**

---

# Project Structure

```
hinglish-bert-classifier/

├── sms_dataset.csv
├── train.py
├── evaluate.py
├── predict.py
├── requirements.txt
├── README.md
└── model_final/
    ├── config.json
    ├── pytorch_model.bin
    └── tokenizer_config.json
```

---

# Ethical Framework

This project follows responsible AI principles.

- Dataset contains only synthetic or anonymized data.
- Model is intended exclusively for defensive cybersecurity.
- Training was performed in an isolated environment.
- Known limitations are documented.
- Designed with compliance to the Indian IT Act 2000 and the Digital Personal Data Protection (DPDP) Act 2023.

---

# Limitations

- Performance is strongest on Hinglish SMS similar to the training dataset.
- Very short SMS messages (<10 words) may receive lower confidence scores.
- Emerging phishing strategies may require periodic retraining.
- Some English-only transactional messages receive lower confidence despite correct classification.

---

# Future Work

- REST API deployment
- Mobile application integration
- Additional Indian regional languages
- Continuous learning from new phishing campaigns
- Confidence threshold optimization
- Larger and more diverse datasets

---

# License

This project is licensed under the MIT License.

See the LICENSE file for details.

---

# References

- HuggingFace Transformers
  https://huggingface.co/transformers/

- DistilBERT Documentation
  https://huggingface.co/docs/transformers/model_doc/distilbert

- Digital Personal Data Protection (DPDP) Act 2023
  https://www.meity.gov.in/
