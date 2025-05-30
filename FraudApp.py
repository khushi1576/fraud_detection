import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import ttkbootstrap as ttk


class FraudDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fraud Detection App")
        self.root.geometry("1020x850")
        self.fraud_data = None
        self.canvas_widget = None

        try:
            self.model = joblib.load("fraud_model.pkl")
        except Exception as e:
            messagebox.showerror("Error", f"Model load failed: {e}")
            return

        self.create_ui()

    def create_ui(self):
        ttk.Label(self.root, text="💸 Fraud Detection System", font=("Helvetica", 18, "bold"),
                  foreground="orange").pack(pady=10)

        ttk.Label(self.root, text="🔹 Enter transaction data (comma-separated):").pack(pady=5)
        self.input_entry = ttk.Entry(self.root, width=80)
        self.input_entry.pack(pady=5)

        self.prediction_label = ttk.Label(self.root, text="Prediction: N/A", font=("Helvetica", 12, "bold"))
        self.prediction_label.pack(pady=5)

        ttk.Button(self.root, text="🧠 Predict Transaction", style="info.TButton",
                   command=self.predict_single_transaction).pack(pady=10)

        ttk.Button(self.root, text="📁 Upload CSV", style="info.TButton", command=self.upload_csv).pack(pady=5)

        self.result_box = tk.Text(self.root, height=12, width=110)
        self.result_box.pack(pady=10)
        self.result_box.tag_configure("alert", foreground="red")
        self.result_box.tag_configure("success", foreground="green")

        self.download_button = ttk.Button(self.root, text="⬇️ Download Fraud Data",
                                          style="primary.TButton", command=self.download_fraud_data)
        self.download_button.pack(pady=5)
        self.download_button.pack_forget()

    def predict_single_transaction(self):
        raw = self.input_entry.get().strip()
        if raw:
            try:
                data = [list(map(float, raw.split(',')))]
                pred = self.model.predict(data)
                label = "Fraudulent" if pred[0] == 1 else "Non-Fraudulent"
                self.prediction_label.config(text=f"Prediction: {label}")

                # Set label color
                if pred[0] == 1:
                    self.prediction_label.config(foreground="red")
                    self.result_box.insert(tk.END, f"Prediction: {label}\n", "alert")
                else:
                    self.prediction_label.config(foreground="green")
                    self.result_box.insert(tk.END, f"Prediction: {label}\n", "success")

                self.result_box.yview(tk.END)
            except Exception as e:
                messagebox.showerror("Error", f"Invalid input: {e}")
        else:
            messagebox.showwarning("Input Error", "Please enter valid data.")

    def upload_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return
        try:
            df = pd.read_csv(file_path)
            X = df.drop(['Class'], axis=1)
            y_pred = self.model.predict(X)
            df['Prediction'] = y_pred
            self.fraud_data = df[df['Prediction'] == 1]

            self.result_box.delete(1.0, tk.END)
            self.result_box.insert(tk.END, f"🚨 {len(self.fraud_data)} Fraudulent Transactions:\n\n", "alert")
            for idx in self.fraud_data.index:
                amount = self.fraud_data.at[idx, 'Amount']
                time = self.fraud_data.at[idx, 'Time']
                self.result_box.insert(tk.END, f"• Index: {idx}, Amount: {amount}, Time: {time}\n", "alert")
            self.result_box.yview(tk.END)

            if not self.fraud_data.empty:
                self.download_button.pack(pady=5)

            self.display_chart(df)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to process file: {e}")

    def display_chart(self, df):
        if self.canvas_widget:
            self.canvas_widget.get_tk_widget().destroy()

        fig, ax = plt.subplots(figsize=(8, 4), dpi=100)

        fraud_only = df[df['Prediction'] == 1]
        ax.scatter(fraud_only['Time'], fraud_only['Amount'], color='red', alpha=0.6, s=10)
        ax.set_title("Fraud: Time vs Amount")
        ax.set_xlabel("Time")
        ax.set_ylabel("Amount")

        plt.tight_layout(pad=3.0)

        canvas = FigureCanvasTkAgg(fig, master=self.root)
        canvas.draw()
        self.canvas_widget = canvas
        self.canvas_widget.get_tk_widget().pack(pady=10)

    def download_fraud_data(self):
        if self.fraud_data is not None and not self.fraud_data.empty:
            try:
                self.fraud_data.to_csv("fraud_data.csv", index=False)
                messagebox.showinfo("Saved", "Fraud data saved as fraud_data.csv")
                self.result_box.insert(tk.END, "\n📁 Data saved to fraud_data.csv\n", "success")
                self.result_box.yview(tk.END)
            except Exception as e:
                messagebox.showerror("Error", f"Could not save file: {e}")
        else:
            messagebox.showerror("No Data", "No fraud data available to save.")


if __name__ == "__main__":
    root = ttk.Window(themename="darkly")
    app = FraudDetectionApp(root)
    root.mainloop()



