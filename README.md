# 🌱 Crop Prediction and Recommendation System

A Machine Learning-based Crop Prediction and Recommendation System that predicts the most suitable crop based on soil and environmental conditions.

The project implements and compares **Logistic Regression, Linear Support Vector Machine (SVM), and a Neural Network (MLPClassifier)** for multi-class crop classification.

---

## 📌 Project Overview

Selecting a suitable crop based on soil nutrients and environmental conditions can help improve agricultural decision-making.

This project uses Machine Learning to predict the appropriate crop using the following parameters:

* Nitrogen (N)
* Phosphorus (P)
* Potassium (K)
* Temperature
* Humidity
* Soil pH
* Rainfall

The system processes the input data, standardizes the features, trains multiple classification models, and predicts the most suitable crop.

---

## 📊 Dataset

The project uses the `Crop_recommendation.csv` dataset.

### Features

| Feature     | Description                |
| ----------- | -------------------------- |
| N           | Nitrogen content in soil   |
| P           | Phosphorus content in soil |
| K           | Potassium content in soil  |
| temperature | Temperature                |
| humidity    | Relative humidity          |
| ph          | Soil pH                    |
| rainfall    | Rainfall                   |

### Target

**label** — The crop recommended for the given soil and environmental conditions.

---

## 🤖 Machine Learning Models

Three classification approaches were implemented.

### 1. Logistic Regression

Logistic Regression is used as a multi-class classification model to predict the crop label.

The model is trained after standardizing the input features using `StandardScaler`.

The trained model is saved as:

```text
lr_model.pkl
```

### 2. Linear Support Vector Machine

A Support Vector Machine with a **linear kernel** is used for crop classification.

```python
svm.SVC(kernel='linear')
```

The trained SVM model and scaler are saved using Joblib:

```text
svm_model.pkl
scaler.pkl
```

### 3. Neural Network

A Multi-Layer Perceptron classifier is used as the Neural Network model.

```python
MLPClassifier(
    hidden_layer_sizes=(100,),
    max_iter=1000,
    random_state=42
)
```

The model uses one hidden layer containing 100 neurons.

Training and testing accuracy are also calculated to evaluate the model's performance.

---

## 📈 Model Performance

The models were evaluated using a train-test split with **80% training data and 20% testing data**.

| Model               |   Accuracy |
| ------------------- | ---------: |
| Logistic Regression | **97.27%** |
| Linear SVM          | **98.18%** |
| Neural Network      | **99.32%** |



---

## 🔄 Machine Learning Workflow

```text
Crop Dataset
     ↓
Feature Selection
     ↓
Train-Test Split
     ↓
StandardScaler
     ↓
Model Training
     ↓
Prediction
     ↓
Model Evaluation
     ↓
Crop Recommendation
```

The same general preprocessing workflow is used for the classification models, with the input features standardized before training and prediction.

---

## 📏 Evaluation Metrics

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

For the SVM and Logistic Regression models, the evaluation metrics are calculated using weighted averaging:

```python
average='weighted'
```

The generated prediction results and evaluation metrics are stored in the `results/` directory.

```text
results/
├── RegressionMetrics.csv
├── SVMMetrics.csv
├── resultRegression.csv
└── resultSVM.csv
```

---

## 🔮 Crop Prediction

The prediction function accepts a user's agricultural input and converts it into a two-dimensional NumPy array before applying the same feature scaling process.

Example input:

```python
sample_input = [
    90,
    42,
    43,
    20.87,
    82.00,
    6.5,
    202.9
]
```

The system can then generate predictions using both:

* SVM
* Logistic Regression

The trained models are saved using Joblib and can be loaded later for prediction.

---

## 💾 Saved Models

The project saves the trained Machine Learning models using Joblib.

```text
lr_model.pkl       → Logistic Regression model
svm_model.pkl      → Linear SVM model
scaler.pkl         → StandardScaler
```

These files can be loaded later without retraining the models.

---

## 📁 Project Structure

```text
Crop_prediction/
│
├── Crop_prediction.ipynb
├── gui.ipynb
├── Crop_recommendation.csv
│
├── lr_model.pkl
├── svm_model.pkl
├── scaler.pkl
│
├── results/
│   ├── RegressionMetrics.csv
│   ├── SVMMetrics.csv
│   ├── resultRegression.csv
│   └── resultSVM.csv
│
└── .gitignore
```

---

## 🛠️ Technologies Used

* **Python**
* **Pandas** — Data loading and manipulation
* **NumPy** — Numerical operations
* **Scikit-learn** — Machine Learning models and evaluation
* **Joblib** — Model serialization
* **Matplotlib** — Visualization
* **Jupyter Notebook** — Development and experimentation

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Jaishritha/Crop_prediction.git
```

### 2. Navigate to the project directory

```bash
cd Crop_prediction
```

### 3. Install the required libraries

```bash
pip install pandas numpy matplotlib scikit-learn joblib jupyter
```

### 4. Open the notebook

```bash
jupyter notebook
```

Open:

```text
Crop_prediction.ipynb
```

Run the notebook cells to train the models, evaluate their performance, and generate predictions.

---

## 🎯 Key Learning Outcomes

This project provided hands-on experience with:

* Data preprocessing
* Feature selection
* Train-test splitting
* Feature standardization
* Multi-class classification
* Logistic Regression
* Support Vector Machines
* Neural Networks
* Model evaluation
* Accuracy, precision, recall and F1-score
* Confusion matrices
* Model persistence using Joblib
* Generating crop predictions
* Comparing different Machine Learning models

---

## 🔮 Future Improvements

Possible improvements include:

* Hyperparameter tuning for the individual models
* Cross-validation for more robust performance evaluation
* Testing additional Machine Learning algorithms
* Improving the user interface
* Deploying the trained model as a web application
* Integrating real-time agricultural or weather data
* Providing additional fertilizer recommendations based on soil conditions

