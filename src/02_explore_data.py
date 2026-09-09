import pandas as pd
from pathlib import Path

# Find the dataset
project_folder = Path(__file__).parent.parent
files = list((project_folder / "dataset").rglob("Sensorless_drive_diagnosis.txt"))

file_path = files[0]

# Load dataset
data = pd.read_csv(file_path, sep=r"\s+", header=None)

print("Dataset loaded successfully!")
print()

# Basic information
print("Dataset shape:")
print(data.shape)

print()

# Check classes
print("Classes in the dataset:")
print(data[48].unique())

print()

# Number of samples for each class
print("Number of samples in each class:")
print(data[48].value_counts().sort_index())

print()

# Check missing values
print("Missing values:")
print(data.isnull().sum().sum())

print()

# Feature statistics
print("Feature statistics:")
print(data.iloc[:, 0:48].describe())