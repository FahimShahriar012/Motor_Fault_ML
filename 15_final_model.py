import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# Find dataset
project_folder = Path(__file__).parent.parent

files = list(
    (project_folder / "dataset").rglob("Sensorless_drive_diagnosis.txt")
)

file_path = files[0]


# Load dataset
data = pd.read_csv(
    file_path,
    sep=r"\s+",
    header=None
)


# Features and target
X = data.iloc[:, 0:48]
y = data.iloc[:, 48]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Final optimized model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("random_forest", RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    ))
])


# Train
print("Training final optimized Random Forest...")
model.fit(X_train, y_train)

print("Training completed!")


# Prediction
print("Making predictions...")
y_pred = model.predict(X_test)


# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print()
print("=" * 50)
print("FINAL MODEL RESULTS")
print("=" * 50)

print()
print(f"Test Accuracy: {accuracy * 100:.6f}%")


# Classification report
print()
print("Classification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        digits=6
    )
)


# Confusion matrix
print()
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save final model
model_path = project_folder / "motor_fault_model.pkl"

joblib.dump(model, model_path)

print()
print("Final model saved successfully!")
print(f"Model location: {model_path}")