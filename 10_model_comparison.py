import matplotlib.pyplot as plt

# Model results
models = [
    "Random Forest",
    "SVM",
    "Logistic Regression",
    "KNN"
]

accuracies = [
    99.897,
    96.744,
    91.942,
    82.875
]

# Create bar chart
plt.figure(figsize=(10, 6))

bars = plt.bar(models, accuracies)

plt.title("Machine Learning Model Accuracy Comparison")
plt.xlabel("Machine Learning Model")
plt.ylabel("Accuracy (%)")

plt.ylim(75, 101)

# Add accuracy values above bars
for bar, accuracy in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.2,
        f"{accuracy:.3f}%",
        ha="center"
    )

plt.tight_layout()

# Save graph
plt.savefig(
    "model_comparison.png",
    dpi=300
)

plt.show()

print("Model comparison graph saved successfully!")