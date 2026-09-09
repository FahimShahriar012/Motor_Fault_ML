import pandas as pd
from pathlib import Path

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


# Find project folder
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


# Create pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    ))
])


# Parameters to test
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 20],
    "model__min_samples_split": [2, 5]
}


# 5-fold stratified CV
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


print("Starting corrected Random Forest tuning...")
print("Testing 8 parameter combinations with 5-fold CV.")
print("This may take some time.")


# Grid search
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)


grid_search.fit(X, y)


# Results
print()
print("Tuning completed!")

print()
print("Best Parameters:")
print(grid_search.best_params_)

print()
print("Best Cross-Validation Accuracy:")
print(f"{grid_search.best_score_ * 100:.4f}%")

print()
print("Best Model:")
print(grid_search.best_estimator_)