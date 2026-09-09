import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


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


# Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Create KNN model
model = KNeighborsClassifier(
    n_neighbors=5
)


# Train
print("Training KNN...")

model.fit(X_train_scaled, y_train)

print("Training completed!")


# Prediction
print("Making predictions...")

y_pred = model.predict(X_test_scaled)


# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print()
print("KNN Accuracy:")
print(accuracy)

print()
print("Accuracy (%):")
print(accuracy * 100)

print()
print("Classification Report:")
print(classification_report(y_test, y_pred))