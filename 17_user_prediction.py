import joblib
import numpy as np
from pathlib import Path


# Project folder
project_folder = Path(__file__).parent.parent

# Model path
model_path = project_folder / "motor_fault_model.pkl"

# Dataset path
dataset_files = list(
    (project_folder / "dataset").rglob("Sensorless_drive_diagnosis.txt")
)

dataset_path = dataset_files[0]


# Load model
model = joblib.load(model_path)


print("=" * 60)
print("           MOTOR FAULT PREDICTION SYSTEM")
print("=" * 60)

print()
print("Model loaded successfully!")

print()
print("Dataset contains 58,509 samples.")
print("Choose a sample number from 1 to 58,509.")

print()


# Get sample number
while True:
    try:
        sample_number = int(input("Enter sample number: "))

        if 1 <= sample_number <= 58509:
            break

        print("Please enter a number between 1 and 58509.")

    except ValueError:
        print("Please enter a valid integer.")


# Load selected sample
data = np.loadtxt(dataset_path)

sample = data[sample_number - 1, :48]

actual_class = int(data[sample_number - 1, 48])


# Reshape sample
sample = sample.reshape(1, -1)


# Prediction
prediction = model.predict(sample)

predicted_class = int(prediction[0])


# Prediction confidence
probabilities = model.predict_proba(sample)

confidence = np.max(probabilities) * 100


# Display result
print()
print("=" * 60)
print("                 PREDICTION RESULT")
print("=" * 60)

print()
print(f"Sample Number:         {sample_number}")
print(f"Actual Motor Class:    {actual_class}")
print(f"Predicted Motor Class: {predicted_class}")
print(f"Prediction Confidence: {confidence:.2f}%")

print()

if actual_class == predicted_class:
    print("Prediction: CORRECT")
else:
    print("Prediction: INCORRECT")

print()
print("=" * 60)