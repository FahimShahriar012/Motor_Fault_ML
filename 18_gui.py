import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np
from pathlib import Path


# ==========================================
# PROJECT PATHS
# ==========================================

project_folder = Path(__file__).parent.parent

model_path = project_folder / "motor_fault_model.pkl"

dataset_files = list(
    (project_folder / "dataset").rglob(
        "Sensorless_drive_diagnosis.txt"
    )
)

dataset_path = dataset_files[0]


# ==========================================
# LOAD MODEL AND DATASET
# ==========================================

model = joblib.load(model_path)

data = np.loadtxt(dataset_path)

total_samples = len(data)


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_fault():

    try:
        sample_number = int(sample_entry.get())

        if sample_number < 1 or sample_number > total_samples:
            messagebox.showerror(
                "Invalid Sample",
                f"Please enter a number between 1 and {total_samples}."
            )
            return

        # Get sensor features
        sample = data[sample_number - 1, :48]

        # Actual class
        actual_class = int(data[sample_number - 1, 48])

        # Reshape
        sample = sample.reshape(1, -1)

        # Prediction
        prediction = model.predict(sample)

        predicted_class = int(prediction[0])

        # Confidence
        probabilities = model.predict_proba(sample)

        confidence = np.max(probabilities) * 100

        # Display results
        actual_value.config(
            text=str(actual_class)
        )

        predicted_value.config(
            text=str(predicted_class)
        )

        confidence_value.config(
            text=f"{confidence:.2f}%"
        )

        if actual_class == predicted_class:

            status_value.config(
                text="✓ CORRECT PREDICTION"
            )

        else:

            status_value.config(
                text="✗ INCORRECT PREDICTION"
            )

    except ValueError:

        messagebox.showerror(
            "Invalid Input",
            "Please enter a valid sample number."
        )


# ==========================================
# MAIN WINDOW
# ==========================================

window = tk.Tk()

window.title("Motor Fault Diagnosis System")

window.geometry("700x550")

window.resizable(False, False)


# ==========================================
# TITLE
# ==========================================

title = tk.Label(
    window,
    text="MOTOR FAULT DIAGNOSIS SYSTEM",
    font=("Arial", 22, "bold")
)

title.pack(pady=25)


subtitle = tk.Label(
    window,
    text="Machine Learning Based Motor Fault Classification",
    font=("Arial", 11)
)

subtitle.pack()


# ==========================================
# INPUT SECTION
# ==========================================

input_frame = tk.Frame(window)

input_frame.pack(pady=30)


sample_label = tk.Label(
    input_frame,
    text="Enter Sample Number:",
    font=("Arial", 13)
)

sample_label.grid(
    row=0,
    column=0,
    padx=10
)


sample_entry = tk.Entry(
    input_frame,
    font=("Arial", 13),
    width=15
)

sample_entry.grid(
    row=0,
    column=1,
    padx=10
)

sample_entry.insert(
    0,
    "100"
)


predict_button = tk.Button(
    window,
    text="PREDICT FAULT",
    font=("Arial", 14, "bold"),
    command=predict_fault,
    width=20,
    height=2
)

predict_button.pack(
    pady=10
)


# ==========================================
# RESULT SECTION
# ==========================================

result_frame = tk.Frame(window)

result_frame.pack(pady=25)


# Actual class

tk.Label(
    result_frame,
    text="Actual Motor Class:",
    font=("Arial", 13)
).grid(row=0, column=0, padx=20, pady=8)

actual_value = tk.Label(
    result_frame,
    text="-",
    font=("Arial", 13, "bold")
)

actual_value.grid(row=0, column=1)


# Predicted class

tk.Label(
    result_frame,
    text="Predicted Motor Class:",
    font=("Arial", 13)
).grid(row=1, column=0, padx=20, pady=8)

predicted_value = tk.Label(
    result_frame,
    text="-",
    font=("Arial", 13, "bold")
)

predicted_value.grid(row=1, column=1)


# Confidence

tk.Label(
    result_frame,
    text="Prediction Confidence:",
    font=("Arial", 13)
).grid(row=2, column=0, padx=20, pady=8)

confidence_value = tk.Label(
    result_frame,
    text="-",
    font=("Arial", 13, "bold")
)

confidence_value.grid(row=2, column=1)


# Status

status_value = tk.Label(
    window,
    text="Waiting for prediction...",
    font=("Arial", 15, "bold")
)

status_value.pack(pady=20)


# ==========================================
# FOOTER
# ==========================================

footer = tk.Label(
    window,
    text="Random Forest Model | 99.897% Test Accuracy",
    font=("Arial", 10)
)

footer.pack(
    side="bottom",
    pady=15
)


# ==========================================
# START GUI
# ==========================================

window.mainloop()