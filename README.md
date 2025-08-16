# 📊 Binary Classification of Bank Marketing Dataset

---

<div style="background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); padding: 20px; border-radius: 12px; color:#222; font-weight:bold;">

## 📌 Project Description  
This project implements a **binary classification model** to predict whether a customer will subscribe to a term deposit (yes/no) based on the **Bank Marketing Dataset**.  
The pipeline covers **data preprocessing, feature engineering, EDA, model building, evaluation, and deployment** from scratch.

</div>

---

<div style="background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%); padding: 20px; border-radius: 12px; color:#222; font-weight:bold;">

## ❓ Problem Statement  
Banks often struggle to identify customers for term deposits.  

**Objective:** Predict customer subscription behavior to **optimize marketing campaigns**, reduce costs, and improve efficiency.  

This project builds a **robust ML pipeline** that prioritizes **Recall** — reducing **False Negatives (FN)** to capture potential customers.

</div>

---

<div style="background: linear-gradient(135deg, #fbc2eb 0%, #a6c1ee 100%); padding: 20px; border-radius: 12px; color:#222; font-weight:bold;">

## 📂 Dataset  
- **Source:** [Bank Marketing Dataset (Kaggle)](https://www.kaggle.com/datasets/sushant097/bank-marketing-dataset-full)  
- **Task:** Predict if a customer subscribes to a term deposit (`yes`/`no`)  
- **Type:** Binary classification  

</div>

---

<div style="background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%); padding: 20px; border-radius: 12px; color:#222; font-weight:bold;">

## ⚙️ Setup Instructions  

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

<div style="background: linear-gradient(135deg, #fddb92 0%, #d1fdff 100%); padding: 20px; border-radius: 12px; color:#222; font-weight:bold;">

## 📦 Libraries Used

* **pandas, numpy** → Data handling & preprocessing  
* **matplotlib, seaborn** → EDA & visualization  
* **scikit-learn** → ML models, metrics, preprocessing  
* **imbalanced-learn** → SMOTE for class imbalance  
* **xgboost, catboost** → Gradient boosting models  
* **joblib** → Model persistence  
* **flask, gunicorn** → Deployment setup  

</div>

---

<div style="background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%); padding: 20px; border-radius: 12px; color:#222; font-weight:bold;">

## 🛠️ Approach Used

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

<div style="background: linear-gradient(135deg, #ffe5b4 0%, #ff9933 100%); padding: 20px; border-radius: 12px; color:#222; font-weight:bold;">

## 📊 Results

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

<div style="background: linear-gradient(135deg, #f6d365 0%, #fda085 100%); padding: 20px; border-radius: 12px; color:#222; font-weight:bold;">

## 🔍 Findings

* High **Recall (0.74)** prioritized over Precision (0.53).  
* **Why Recall matters:**  
  * **False Negative (FN):** Missed potential customer → costly  
  * **False Positive (FP):** Slight extra marketing → acceptable  
* Most potential customers are captured  
* **Stacked ensemble** with Logistic Regression meta-learner → **ROC AUC = 0.92**  

**Summary:**  
* **Recall ↑** → Capture more subscribers  
* **Precision ↓ (acceptable)** → Extra calls/emails fine  

</div>

---

<div style="background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); padding: 20px; border-radius: 12px; color:#222; font-weight:bold;">

## 👨‍💻 Author

Implemented end-to-end by **Bhaskar Mishra** — covering **data preprocessing, feature engineering, EDA, modeling, evaluation, and deployment pipeline setup**.

</div>
