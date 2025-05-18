import argparse
import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

def main(data_path):

    # Load preprocessed dataset
    df = pd.read_csv(data_path)

    # Pisahkan fitur dan target
    X = df.drop(columns=["performance_level"])
    y = df["performance_level"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Normalisasi fitur numerik
    scaler = StandardScaler()
    numeric_cols = ["math score", "reading score", "writing score", "average_score"]
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # Inisialisasi tracking MLflow
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("student-performance-classification")

    # Dictionary model
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Evaluasi
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(y_test, y_pred)

            # Logging parameter dan metric
            mlflow.log_param("model_type", model_name)
            mlflow.log_metric("accuracy", acc)

            for label in report:
                if label in ["accuracy", "macro avg", "weighted avg"]:
                    continue
                mlflow.log_metric(f"precision_{label}", report[label]["precision"])
                mlflow.log_metric(f"recall_{label}", report[label]["recall"])
                mlflow.log_metric(f"f1_score_{label}", report[label]["f1-score"])

            # Simpan model
            mlflow.sklearn.log_model(model, "model")

            # Plot confusion matrix
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                        xticklabels=np.unique(y), yticklabels=np.unique(y))
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title(f"Confusion Matrix: {model_name}")

            # Simpan gambar dan log artifact
            os.makedirs("images", exist_ok=True)
            fig_path = f"images/cm_{model_name}.png"
            plt.savefig(fig_path)
            mlflow.log_artifact(fig_path)
            plt.close()

    print("Selesai menjalankan semua model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML models on student performance data")
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="students_performance_preprocessed_automated.csv",
        help="Path to preprocessed dataset CSV file"
    )
    args = parser.parse_args()
    main(args.data_path)