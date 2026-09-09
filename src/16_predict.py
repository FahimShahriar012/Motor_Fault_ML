import joblib
import numpy as np
from pathlib import Path


# Project folder
project_folder = Path(__file__).parent.parent

# Load trained model
model_path = project_folder / "motor_fault_model.pkl"

model = joblib.load(model_path)

print("=" * 50)
print("      MOTOR FAULT PREDICTION SYSTEM")
print("=" * 50)

print()
print("Model loaded successfully!")

print()
print("This model requires 48 sensor features.")
print("For testing, we will use one sample from the dataset.")

# Load dataset
dataset_files = list(
    (project_folder / "dataset").rglob("Sensorless_drive_diagnosis.txt")
)

dataset_path = dataset_files[0]

data = np.loadtxt(dataset_path)

# Take one sample
sample = data[0, :48]

# Actual class
actual_class = int(data[0, 48])

# Reshape for prediction
sample = sample.reshape(1, -1)

# Prediction
prediction = model.predict(sample)

predicted_class = int(prediction[0])

print()
print("=" * 50)
print("PREDICTION RESULT")
print("=" * 50)

print()
print(f"Actual Motor Class:    {actual_class}")
print(f"Predicted Motor Class: {predicted_class}")

if actual_class == predicted_class:
    print()
    print("Prediction: CORRECT")
else:
    print()
    print("Prediction: INCORRECT")

print()
print("=" * 50)