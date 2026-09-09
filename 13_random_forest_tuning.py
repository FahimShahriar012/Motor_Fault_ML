import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
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


# Create pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    ))
])


# Hyperparameters to test
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 20],
    "model__min_samples_split": [2, 5]
}


# Grid Search
print("Starting Random Forest Hyperparameter Tuning...")
print("This may take some time.")


grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)


grid_search.fit(X, y)


# Results
print()
print("Hyperparameter tuning completed!")

print()
print("Best Parameters:")
print(grid_search.best_params_)

print()
print("Best Cross-Validation Accuracy:")
print(grid_search.best_score_)

print()
print("Best Accuracy (%):")
print(grid_search.best_score_ * 100)