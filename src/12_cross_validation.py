import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


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


# Separate features and target
X = data.iloc[:, 0:48]
y = data.iloc[:, 48]


# Create ML pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])


# 5-fold stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


print("Starting 5-Fold Cross-Validation...")
print("This may take some time.")


# Calculate scores
scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)


# Display results
print()
print("Cross-Validation Results:")

for i, score in enumerate(scores, start=1):
    print(f"Fold {i}: {score * 100:.4f}%")


print()
print(f"Mean Accuracy: {scores.mean() * 100:.4f}%")
print(f"Standard Deviation: {scores.std() * 100:.4f}%")