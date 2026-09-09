import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

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

print("Train-test split completed!")
print()

print("Training data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print()

print("Testing data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)