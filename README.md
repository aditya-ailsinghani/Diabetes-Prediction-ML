# Diabetes-Prediction-ML
A classification model to predict diabetes based on patient medical attributes using Logistic Regression, Random Forest, Decision Trees, and SVM. Includes data preprocessing, feature selection (RFE), class balancing (SMOTE), and model evaluation (Precision-Recall, ROC-AUC curves, Confusion Matrices).
🚀 Best Model: Logistic Regression (AUC = 0.82)
📊 Tech Stack: Python, Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn

---

## 📌 Features:
✅ [**Data Preprocessing**]  
✅ [**Feature Selection using Recursive Feature Elimination (RFE)**] 
✅ [**SMOTE for Class Balancing**]  
✅ [**Trained Multiple Models**]  
✅ [**Model Performance Evaluation**]

---

## 📊 Dataset:
📌 **Pima Indians Diabetes Database**  
📂 **Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
- **Contains:** 768 patient records with 8 medical attributes  
- **Target Variable:** `Outcome` (1 = Diabetic, 0 = Non-Diabetic)

---

## **🛠 Data Preprocessing**
🔹 Handled **missing values** using **median imputation**.  
🔹 Removed **outliers in Insulin** using the **IQR method**.  
🔹 **Standardized features** using **MinMax Scaling**.

---

## **📊 Exploratory Data Analysis (EDA)**  
🔹 **Heatmap of feature correlations** to check feature importance.
![Heatplot](images/Heatplot.png)

🔹 **Plotted distributions and boxplots** to identify skewness and outliers.
![OutcomeVariable](images/OutcomeVariable.png)
![Pairplot](images/Pairplot.png)
![BoxPlot](images/BoxPlot.png)


---

## **🔍 Feature Selection using RFE**
✅ **What is RFE?**  
- Recursive Feature Elimination (RFE) **removes less important features** iteratively.
- Improves **model performance** by reducing noise.

---

✅ **Why RFE?**  
- Selecting **top 5 features** helps in generalization.
- Avoids overfitting.
  
---

## **🧠 Model Training & Parameter Tuning**
🔹 **Trained multiple models:**  
- Logistic Regression  
- Decision Tree  
- Random Forest  
- SVM
  
---

🔹 **Hyperparameter tuning using GridSearchCV**  
- **Logistic Regression:** Tuned `C`, `max_iter`.  
- **Decision Tree & Random Forest:** Tuned `max_depth`, `min_samples_split`.

---
![image](https://github.com/user-attachments/assets/053c6545-7f55-4081-8ddd-6ba8b2252a81)

---
## **📊 Results & Performance Metrics**
🔹 **Confusion Matrix** to analyze True Positives & False Negatives.  
🔹 **Precision-Recall & ROC-AUC curves** for model evaluation.  
![image](https://github.com/user-attachments/assets/37ab7aaf-315d-41e7-9235-4bc566c4a342)
![image](https://github.com/user-attachments/assets/89773599-b7fa-4fcd-a4f8-8121e18fc624)

📌 **Best Model:** **Logistic Regression (AUC = 0.82), Random Forest (AUC = 0.80)** 


---

## **📌 Conclusion**
✅ **Best Model:** Logistic Regression (AUC = 0.82).  
✅ **Feature selection (RFE) improved accuracy**.  
✅ **Using SMOTE helped in balancing dataset**.  

📌 **Future Work:**  
- Experiment with **XGBoost, Deep Learning** to improve model accuracy

---

## 📜 License
This project is **open-source** under the **MIT License**.

---
