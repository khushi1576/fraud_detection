# Fraud Detection System

A Python-based fraud detection application using machine learning (Random Forest classifier) for detecting fraudulent credit card transactions. This project includes a trained model, a Tkinter GUI app for prediction, and exploratory data analysis.

---

## Features

- Train a Random Forest model on the credit card fraud dataset
- Evaluate model performance with accuracy, precision, recall, F1-score, MCC, and confusion matrix
- GUI application with Tkinter to:
  - Predict fraud on single transaction input
  - Upload CSV files and detect frauds in bulk
  - Visualize fraud data with scatter plot (Time vs Amount)
  - Download fraud transaction data as CSV
- Clean and modular Python codebase

---

## Files in this Repository

- `fraudmodel.py` — Core model training, evaluation, saving model & metrics  
- `fraudapp.py` — Tkinter GUI for fraud detection  
- `creditcard.ipynb` — Data exploration, preprocessing, and model experimentation (Jupyter notebook)  
- `requirements.txt` — Python dependencies  
- `.gitignore` — Files and folders to ignore in Git  
- `README.md` — Project overview and instructions  

---

## Installation

1. Clone the repo:

```bash
git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection
