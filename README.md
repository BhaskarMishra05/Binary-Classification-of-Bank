<div style="background: linear-gradient(to right, #8e2de2, #4a00e0); padding: 20px; border-radius: 10px; color: white;">

# 🏦 **Binary Classification of Bank Marketing Dataset**

</div>

---

<div style="background: linear-gradient(to right, #6a11cb, #2575fc); padding: 15px; border-radius: 10px; color: white;">

## 📌 **Project Description**  
This project implements a **binary classification model** to predict whether a customer will subscribe to a term deposit (yes/no) based on the **Bank Marketing Dataset**.  
The pipeline includes **data preprocessing, feature engineering, EDA, model building, evaluation, and deployment setup** from scratch.

</div>

---

<div style="background: linear-gradient(to right, #ff416c, #ff4b2b); padding: 15px; border-radius: 10px; color: white;">

## ❓ **Problem Statement**  
Financial institutions often struggle with identifying the right customers for term deposits.  

**Objective:** Predict customer subscription behavior to **optimize marketing campaigns**, reduce costs, and improve efficiency.  

This project builds a **robust ML pipeline** that prioritizes **Recall** — reducing **False Negatives (FN)** to capture potential customers.

</div>

---

<div style="background: linear-gradient(to right, #43cea2, #185a9d); padding: 15px; border-radius: 10px; color: white;">

## 📂 **Dataset**  
- **Source:** [Bank Marketing Dataset (Kaggle)](https://www.kaggle.com/datasets/sushant097/bank-marketing-dataset-full)  
- **Task:** Predict if a customer subscribes to a term deposit (`yes`/`no`)  
- **Type:** Binary classification  

</div>

---

<div style="background: linear-gradient(to right, #ffafbd, #ffc3a0); padding: 15px; border-radius: 10px; color: white;">

## ⚙️ **Setup Instructions**  

1. Clone the repository:  
```bash
git clone https://github.com/BhaskarMishra05/Binary-Classification-of-Bank.git
cd Binary-Classification-of-Bank
```
2. Create and activate a virtual environment.
3. Install dependencies:  
```bash
pip install -r requirements.txt
```
4. Run the pipeline or application.

</div>

---

<div style="background: linear-gradient(to right, #fc4a1a, #f7b733); padding: 15px; border-radius: 10px; color: white;">

## 📦 **Libraries Used**

* **pandas, numpy** → Data handling & preprocessing  
* **matplotlib, seaborn** → EDA & visualization  
* **scikit-learn** → ML models, metrics, preprocessing  
* **imbalanced-learn** → SMOTE for class imbalance  
* **xgboost, catboost** → Gradient boosting models  
* **joblib** → Model persistence  
* **flask, gunicorn** → Deployment setup  

</div>

---

<div style="background: linear-gradient(to right, #8360c3, #2ebf91); padding: 15px; border-radius: 10px; color: white;">

## 🛠️ **Approach Used**

**Feature Engineering & Preprocessing:**  
* Ordinal encoding for `education` and `job`  
* Mapping binary columns (`default`, `loan`, `housing`) to 0/1  
* Cyclical encoding for `month` and `day`  
* Derived `dept` column based on account balance  
* Imputation & scaling for numeric, OneHotEncoding for categorical

**Resampling:** Applied **SMOTE** to handle class imbalance  

**Base Learners (Stacked Models):**  
* **XGBoost**  
* **AdaBoostClassifier**  
* **HistGradientBoostingClassifier** (`class_weight='balanced'`)  
* **ExtraTreesClassifier** (`class_weight='balanced'`)  
* **CatBoostClassifier**  

**Meta Learner (Final Estimator):**  
* **Logistic Regression** with class balancing  

**Pipeline:** **SMOTE + StackingClassifier**  

**Threshold Adjustment:** **0.30 probability cutoff** to focus on Recall  

</div>

---

<div style="background: linear-gradient(to right, #36d1dc, #5b86e5); padding: 15px; border-radius: 10px; color: white;">

## 📊 **Results**

| Metric        | Value  |
| ------------- | ------ |
| **Accuracy**  | 0.8914 |
| **Precision** | 0.5364 |
| **Recall**    | 0.7369 |
| **F1 Score**  | 0.6208 |
| **ROC AUC**   | 0.9228 |

**Confusion Matrix:**

```
[[7257  695]
 [ 287  804]]
```

</div>

---

<div style="background: linear-gradient(to right, #ee9ca7, #ffdde1); padding: 15px; border-radius: 10px; color: white;">

## 🔍 **Findings**

* High **Recall (0.74)** was prioritized over Precision (0.53) intentionally.  
* **Why Recall matters more**:  
  * **False Negative (FN):** Missed potential customer → costly for bank  
  * **False Positive (FP):** Slightly extra marketing cost → acceptable  
* Ensures most potential customers are captured  
* **Stacked ensemble** with Logistic Regression as meta-learner achieved **ROC AUC = 0.92**  

**Summary:**  
* **Recall ↑** → Capture more subscribers  
* **Precision ↓ (acceptable)** → Extra calls/emails are fine  

</div>

---

<div style="background: linear-gradient(to right, #fc00ff, #00dbde); padding: 15px; border-radius: 10px; color: white;">

## 👨‍💻 **Author**

Implemented end-to-end by **Bhaskar Mishra** — covering **data preprocessing, feature engineering, EDA, modeling, evaluation, and deployment pipeline setup**.

</div>
