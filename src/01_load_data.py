import pandas as pd
from pathlib import Path

# Find the dataset file automatically
project_folder = Path(__file__).parent.parent
files = list((project_folder / "dataset").rglob("Sensorless_drive_diagnosis.txt"))

if not files:
    print("Dataset file was not found!")
else:
    file_path = files[0]

    print("Dataset found at:")
    print(file_path)
    print()

    data = pd.read_csv(file_path, sep=r"\s+", header=None)

    print("Dataset loaded successfully!")
    print()

    print("Dataset shape:")
    print(data.shape)

    print()
    print("First 5 rows:")
    print(data.head())