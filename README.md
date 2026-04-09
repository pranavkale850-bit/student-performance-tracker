# student-performance-tracker
ML project to predict student performance and provide improvement recommendations using feature analysis and classification models.

# 📊 Student Performance Tracker

## 🚀 Student Performance Analysis & Recommendation System

---

## 📌 Overviewnan

This project predicts student academic performance and provides actionable recommendations using machine learning techniques.

---

## 🎯 Objectives

* Analyze factors affecting student performance
* Build a predictive ML model
* Identify key features impacting results
* Provide insights for improvement

---

## 🧠 Features

* Data preprocessing & cleaning
* Feature importance analysis
* Performance prediction using ML models
* Model evaluation and comparison

---

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

---

## 📂 Project Structure

```
student-performance-tracker/
│
├── data/
├── src/
│   ├── data_analysis.py
│   └── model_training.py
│
├── README.md
└── requirements.txt
```

---

## 📊 Results (Current)

* Accuracy: ~48–50%
* Key Features:

  * absences
  * age (unexpected ⚠️)
  * health
  * goout
  * Walc

---

## 📅 Progress Log

### Day 4

* Uploaded project to GitHub
* Feature importance analysis completed
* Identified key influencing factors
* Accuracy drop observed

---

## ⚠️ Challenges

* Age feature behaving unexpectedly
* Accuracy instability
* Feature selection issues

---

## 🎯 Future Improvements

* Better feature engineering
* Hyperparameter tuning
* Try advanced models (XGBoost, Logistic Regression)

---

## 🚀 Day 5 Progress

### 🔹 Handling Class Imbalance
- Applied `class_weight="balanced"` in Random Forest
- Improved detection of minority class ("Good" students)

### 🔹 Model Improvements
- Compared Logistic Regression, Decision Tree, and Random Forest
- Random Forest showed best performance (~51% accuracy)

### 🔹 Feature Importance
- Identified key factors affecting performance:
  - Absences
  - Study habits (goout, Walc)
  - Health
- Observed that some features (e.g., age) had indirect influence

### 🔹 Recommendation System
- Built rule-based recommendation logic
- Generates actionable suggestions based on student behavior
- Example:
  - Low study time → suggest increase
  - High absences → suggest reduction

### 🔹 Key Insight
> Model predictions were enhanced by combining machine learning with rule-based recommendations to create a practical student support system.



## 🧠 Learning Progress

- Learned how to handle class imbalance in ML models
- Understood trade-offs between accuracy and fairness
- Implemented feature importance for model interpretability
- Built a basic recommendation engine using model insights

👨‍💻 Author

~ Pranav Kale
