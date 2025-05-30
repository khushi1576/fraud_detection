# Fraud Detection App

A desktop application to detect fraudulent credit card transactions using a Random Forest classifier trained on the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

---

## Dataset

This project uses the publicly available **Credit Card Fraud Detection** dataset from Kaggle:

[https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

**Note:** You need to download `creditcard.csv` from Kaggle and place it in the project root directory before training or running the app.

---

## Features

- Train a Random Forest model to detect fraud.
- Predict fraud on single transactions from user input.
- Upload CSV files for batch fraud detection.
- Visualize fraudulent transactions with scatter plots.
- Download detected fraud cases as CSV.
- User-friendly GUI built with Tkinter and ttkbootstrap.

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/fraud-detection-app.git
   cd fraud-detection-app
