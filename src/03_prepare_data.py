import pandas as pd
from pathlib import Path

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

print("Data preparation successful!")
print()

print("X shape:")
print(X.shape)

print()

print("y shape:")
print(y.shape)

print()

print("First 5 feature rows:")
print(X.head())

print()

print("First 5 target values:")
print(y.head())