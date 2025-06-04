# 🎯 Student Exam Score Prediction

This project applies machine learning and neural networks to predict final student exam scores based on study habits, previous grades, family background, and other features. It includes data preprocessing, multiple regression models, deep learning, and performance evaluation.

---

## 📌 Goal
Build and evaluate a regression model that accurately predicts student exam scores using features such as:
- Study time
- Prior grades (G1, G2)
- Parental education level
- Absences, failures, internet access, etc.

---

## 🧠 Dataset

### Source
- 📍 UCI ML Repository: [Student Performance Data](https://archive.ics.uci.edu/ml/datasets/Student+Performance)
- 📍 Kaggle: [Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)

### Target Variable
- `G3` (Final exam score) in the UCI dataset  
  or  
- `math score`, `reading score`, or `writing score` in the Kaggle version

---

## 📂 Features Used

- `studytime`: Weekly study time
- `failures`: Number of past class failures
- `absences`: Number of school absences
- `G1`, `G2`: Previous grades
- `school`, `sex`, `age`, `address`, `Pstatus`: Demographics
- `Medu`, `Fedu`: Parental education
- `internet`: Internet access at home

---

## 🔄 Workflow

### 1. Data Analysis & Preprocessing
- Load data with Pandas
- Explore distributions, outliers, and missing values
- Encode categorical variables (OneHot, LabelEncoder)
- Normalize/standardize numerical features
- Split data into training and testing sets (80/20)

### 2. Machine Learning Models
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- K-Nearest Neighbors (KNN) Regressor
- Support Vector Regressor (SVR)

### 3. Neural Network Model
Using **TensorFlow/Keras**:
- Input Layer: Number of selected features
- Hidden Layers: Dense (64 ➝ 32), ReLU activation
- Output Layer: 1 neuron, Linear activation
- Loss Function: MSE (Mean Squared Error)
- Optimizer: Gradient Descent / Adam

### 4. Training & Evaluation
- Compare ML models and Neural Network performance
- Evaluate using:
  - ✅ MAE (Mean Absolute Error)
  - ✅ RMSE (Root Mean Squared Error)
  - ✅ R² Score
- Visualizations:
  - 📈 NN training loss curve
  - 📊 Predictions vs. Actual values
  - 🔍 Feature importance (for tree-based models)

---

## ⭐ Bonus (Optional)
- Git for version control
- Jupyter Notebook for documentation
- Hyperparameter tuning using `GridSearchCV`
- Save models and result plots

---

## 🧪 Tools & Technologies
- Python, Jupyter Notebook
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn, TensorFlow/Keras
- Git & GitHub

---

## 📊 Skills Demonstrated
- Intermediate Python programming
- Data preprocessing & visualization
- ML model training and evaluation
- Building neural networks from scratch
- Gradient descent, MSE, ReLU, normalization
- Git and collaborative coding practices

---

## 📄 License
This project is licensed under the **MIT License** – feel free to use, modify, and share with attribution.

