import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

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

# Scale the features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
print("Training Random Forest...")
model.fit(X_train_scaled, y_train)

print("Training completed!")

# Make predictions
y_pred = model.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print()
print("Model Accuracy:", accuracy)
print("Model Accuracy (%):", accuracy * 100)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

print()
print("Confusion Matrix:")
print(cm)

# Display confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

disp.plot(xticks_rotation="vertical")
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.show()

print()
print("Classification Report:")
print(classification_report(y_test, y_pred))