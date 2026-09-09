# Motor Fault Diagnosis Using Machine Learning

## Overview

This project presents a machine learning-based system for automatic motor fault classification using sensorless drive diagnosis data.

The project evaluates multiple machine learning algorithms and develops an optimized Random Forest model for accurate motor fault classification.

The final system also includes a prediction module and a graphical user interface (GUI).

---

## Project Objectives

- Analyze sensorless motor drive data.
- Preprocess and scale the sensor features.
- Compare different machine learning algorithms.
- Optimize the best-performing model.
- Evaluate model performance using multiple metrics.
- Identify the most important features.
- Develop a motor fault prediction system.
- Develop a graphical user interface for prediction.

---

## Dataset

The project uses the **Sensorless Drive Diagnosis** dataset.

Dataset characteristics:

- Total samples: **58,509**
- Input features: **48**
- Classes: **11**
- Missing values: **0**
- Training samples: **46,807**
- Testing samples: **11,702**

The dataset is stored locally in:

```text
dataset/Sensorless Drive Diagnosis.txt

Machine Learning Models

The following models were evaluated:

Random Forest
Logistic Regression
K-Nearest Neighbors (KNN)
Support Vector Machine (SVM)

Model Performance
| Model               |    Accuracy |
| ------------------- | ----------: |
| Random Forest       | **99.897%** |
| SVM                 |     96.744% |
| Logistic Regression |     91.942% |
| KNN                 |     82.875% |

Random Forest achieved the highest test accuracy.
Final Random Forest Model

The optimized Random Forest model achieved:

Test Accuracy: 99.897453%

The model was evaluated using:

Classification report
Confusion matrix
5-fold cross-validation
Feature importance analysis
Hyperparameter tuning
Cross-Validation

The optimized Random Forest achieved:

Mean 5-Fold Accuracy: 99.8616%

Standard Deviation: 0.0303%

Optimized Parameters
n_estimators = 200
max_depth = None
min_samples_split = 5
Feature Importance

The most important features identified by the Random Forest model include:

Feature 11
Feature 10
Feature 12
Feature 8
Feature 9
Feature 7
Feature 24
Feature 22
Feature 19
Feature 23

Feature 11 had the highest importance among the evaluated features.
Project Structure
Motor_Fault_ML/
│
├── dataset/
│   └── Sensorless Drive Diagnosis.txt
│
├── src/
│   ├── 01_load_data.py
│   ├── 02_explore_data.py
│   ├── 03_prepare_data.py
│   ├── 04_train_test_split.py
│   ├── 05_feature_scaling.py
│   ├── 06_random_forest.py
│   ├── 07_logistic_regression.py
│   ├── 08_knn.py
│   ├── 09_svm.py
│   ├── 10_model_comparison.py
│   ├── 11_feature_importance.py
│   ├── 12_cross_validation.py
│   ├── 13_random_forest_tuning.py
│   ├── 14_corrected_tuning.py
│   ├── 15_final_model.py
│   ├── 16_predict.py
│   ├── 17_user_prediction.py
│   └── 18_gui.py
│
├── motor_fault_model.pkl
└── README.md
Technologies Used
Python
NumPy
Pandas
Scikit-learn
Matplotlib
Tkinter
Joblib

Installation
Clone the repository:
git clone YOUR_GITHUB_REPOSITORY_URL
Navigate to the project directory:
cd Motor_Fault_ML
Install the required Python packages:
pip install numpy pandas scikit-learn matplotlib joblib

Running the Project
Explore the dataset
python src/02_explore_data.py

Train the models
python src/06_random_forest.py
python src/07_logistic_regression.py
python src/08_knn.py
python src/09_svm.py

Evaluate the final model
python src/15_final_model.py

Test a prediction
python src/17_user_prediction.py

Launch the GUI
python src/18_gui.py

GUI

The project includes a graphical interface that allows users to select a sensor-data sample and obtain a motor fault classification using the trained Random Forest model.

The system displays:

Actual motor class
Predicted motor class
Prediction confidence
Prediction status
Results

The final optimized Random Forest classifier demonstrated excellent performance on the held-out test dataset.

Final Test Accuracy: 99.897%

The results indicate that the sensorless drive features contain strong information for distinguishing between the 11 motor classes in this dataset.

Future Work

Future improvements may include:

Real-time motor sensor data acquisition.
Deployment on embedded hardware.
Integration with industrial monitoring systems.
Testing on additional motor datasets.
Deep learning-based fault classification.
Real-time fault alerts.
Cloud-based motor condition monitoring.
Author

Fahim Shahriar

Electrical Engineering and Automation

Disclaimer

This project is intended for academic and research purposes. The reported performance is based on the evaluated dataset and should not be interpreted as guaranteed performance on unseen real-world motor systems.


### Step 2

Save the file:

**Ctrl + S**

Then **don't upload anything to GitHub yet**.

Next, we'll create a `requirements.txt` and a `.gitignore`, then I'll guide you through uploading the entire project to GitHub using PowerShell.
