import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt


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


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Scale features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Train Random Forest
print("Training Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train_scaled, y_train)

print("Training completed!")


# Get feature importance
importance = model.feature_importances_

feature_names = [f"Feature {i+1}" for i in range(48)]

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})


# Sort from highest to lowest
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)


print()
print("Top 10 Most Important Features:")
print(feature_importance.head(10))


# Plot top 10 features
top10 = feature_importance.head(10)

plt.figure(figsize=(10, 6))

plt.barh(
    top10["Feature"][::-1],
    top10["Importance"][::-1]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Most Important Features")

plt.tight_layout()

plt.savefig(
    "feature_importance.png",
    dpi=300
)

plt.show()

print()
print("Feature importance graph saved successfully!")