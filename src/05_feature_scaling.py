import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Find the dataset
project_folder = Path(__file__).parent.parent
files = list(
    (project_folder / "dataset").rglob("Sensorless_drive_diagnosis.txt")
)

file_path = files[0]

# Load dataset
data = pd.read_csv(file_path, sep=r"\s+", header=None)

# Separate features and target
X = data.iloc[:, 0:48]
y = data.iloc[:, 48]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create scaler
scaler = StandardScaler()

# Fit scaler ONLY on training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data using the same scaler
X_test_scaled = scaler.transform(X_test)

print("Feature scaling completed!")
print()

print("Original X_train shape:")
print(X_train.shape)

print()

print("Scaled X_train shape:")
print(X_train_scaled.shape)

print()

print("Scaled X_test shape:")
print(X_test_scaled.shape)

print()

print("First 5 scaled samples:")
print(X_train_scaled[:5])