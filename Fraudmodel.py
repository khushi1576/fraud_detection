import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix
)
import joblib

class FraudDetectionModel:
    def __init__(self, data_path="creditcard.csv"):
        self.data_path = data_path
        self.data = None
        self.model = None
        self.metrics = {}
        self.conf_matrix = None

    def load_data(self):
        try:
            # Load dataset efficiently
            self.data = pd.read_csv(self.data_path)
            print("📂 Data loaded successfully")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
        return self.data

    def preprocess(self):
        # Preprocessing: splitting data into features and target
        self.X = self.data.drop(['Class'], axis=1)
        self.Y = self.data["Class"]
        self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(
            self.X, self.Y, test_size=0.2, random_state=42)
        print("🔧 Data preprocessed: Training and Testing split complete")
        return self.xTrain, self.xTest, self.yTrain, self.yTest

    def train_model(self):
        # Using n_jobs=-1 for parallel processing to speed up training
        self.model = RandomForestClassifier(n_jobs=-1)
        self.model.fit(self.xTrain, self.yTrain)  # Model training
        self.yPred = self.model.predict(self.xTest)  # Predictions

        # Model evaluation metrics
        self.metrics['accuracy'] = accuracy_score(self.yTest, self.yPred)
        self.metrics['precision'] = precision_score(self.yTest, self.yPred)
        self.metrics['recall'] = recall_score(self.yTest, self.yPred)
        self.metrics['f1_score'] = f1_score(self.yTest, self.yPred)
        self.metrics['mcc'] = matthews_corrcoef(self.yTest, self.yPred)
        self.conf_matrix = confusion_matrix(self.yTest, self.yPred)

        print("✅ Model successfully created and trained!")
        return self.metrics

    def get_confusion_matrix(self):
        return self.conf_matrix

    def predict(self, input_data):
        return self.model.predict(input_data)

    def predict_single(self, input_data):
        # Predict single entry (ensure input is in the correct format)
        return self.model.predict([input_data])  # Assumes input_data is a single row

    def save_model(self):
        # Save the trained model to a file
        joblib.dump(self.model, "fraud_model.pkl")
        print("✅ Model saved successfully as fraud_model.pkl")

    def save_metrics(self):
        # Save metrics to a text file
        with open("metrics.txt", "w") as f:
            for k, v in self.metrics.items():
                f.write(f"{k.capitalize()}: {v:.4f}\n")
        print("✅ Metrics saved to metrics.txt")

    def get_data_stats(self):
        # Get statistics for fraud detection
        fraud = self.data[self.data['Class'] == 1]
        valid = self.data[self.data['Class'] == 0]
        outlier_fraction = len(fraud) / float(len(valid))

        stats = {
            "shape": self.data.shape,
            "fraud_cases": len(fraud),
            "valid_cases": len(valid),
            "outlier_fraction": outlier_fraction,
            "fraud_amount_stats": fraud["Amount"].describe().to_dict(),
            "valid_amount_stats": valid["Amount"].describe().to_dict(),
        }
        return stats

# ------------------ AUTO RUN WHEN EXECUTED DIRECTLY ------------------

if __name__ == "__main__":
    model = FraudDetectionModel()
    model.load_data()  # Load data
    model.preprocess()  # Preprocess data
    metrics = model.train_model()  # Train the model
    model.save_model()  # Save the trained model
    model.save_metrics()  # Save the metrics to a file

    print("📊 Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k.capitalize()}: {v:.4f}")
