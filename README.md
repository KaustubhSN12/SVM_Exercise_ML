# ⚡ Support Vector Machine (SVM) Classifier – Social Network Ads Dataset

This repository demonstrates the implementation of the **Support Vector Machine (SVM)** algorithm for binary classification using the **Social Network Ads** dataset.  
The project explores both **RBF kernel** and **Linear kernel** SVMs for predicting user purchase behavior based on **Age** and **Estimated Salary**.

---

## 📌 Overview
- **Algorithm**: Support Vector Machine (SVM)  
- **Dataset**: Social Network Ads (400 records)  
- **Goal**: Predict whether a user purchased a product (1 = Yes, 0 = No)  
- **Features**:  
  - `Age`  
  - `EstimatedSalary`  

---

## 📊 Dataset Information
The dataset contains **400 entries** with the following columns:  

| User ID | Gender | Age | EstimatedSalary | Purchased |
|---------|--------|-----|-----------------|-----------|
| 15624510 | Male   | 19  | 19000           | 0         |
| 15810944 | Male   | 35  | 20000           | 0         |
| 15668575 | Female | 26  | 43000           | 0         |
| 15603246 | Female | 27  | 57000           | 0         |
| 15804002 | Male   | 19  | 76000           | 0         |

- Target variable: **Purchased** (0 = No, 1 = Yes)  
- Balanced with ~35% positive class.

---

## ⚙️ Steps Covered

### 🔹 1. Data Preprocessing
```python
import pandas as pd

data = pd.read_csv("Social_Network_Ads.csv")
X = data.iloc[:, [2, 3]].values   # Age, EstimatedSalary
y = data.iloc[:, 4].values
```

- Selected only **Age** and **EstimatedSalary** for simplicity.  
- Encoded target variable (Purchased).  

---

### 🔹 2. Train-Test Split & Feature Scaling
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

---

### 🔹 3. Model Training (SVM with Kernels)
#### 🔸 RBF Kernel
```python
from sklearn.svm import SVC

classifier = SVC(kernel="rbf", random_state=0)
classifier.fit(X_train, y_train)
```

#### 🔸 Linear Kernel
```python
classifier = SVC(kernel="linear", random_state=0)
classifier.fit(X_train, y_train)
```

---

### 🔹 4. Predictions & Evaluation
```python
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Accuracy:", accuracy_score(y_test, y_pred))
```

📌 Example Output:  
- **Confusion Matrix**:  
  ```
  [[66  2]
   [ 8 24]]
  ```  
- **Accuracy**: 0.90 (90%)  

---

## 🔑 Key Learning Outcomes
- Importance of **feature scaling** for SVMs.  
- How to apply **different kernels (RBF, Linear)** in classification.  
- Evaluating performance using **confusion matrix & accuracy score**.  

---

## 📚 References
- [Scikit-learn SVM Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)  
- [Introduction to SVMs](https://scikit-learn.org/stable/modules/svm.html)  

---

## 🔗 Explore My Other Repositories
- 🌸 [KNN Algorithm Exercise – Iris Dataset](https://github.com/KaustubhSN12/KNN_Algorithm_Exercise_ML)  
- 🤖 [Naive Bayes Algorithm Exercise – Fish Dataset](https://github.com/KaustubhSN12/Naive-bayes-algorithm_ML_Exercise)  
- 🚀 [K-Means Clustering Exercise – Titanic Dataset](https://github.com/KaustubhSN12/Kmeans_Cluster_Exercise_ML)  
- ⚡ [SVM Algorithm Exercise – Social Network Ads](https://github.com/KaustubhSN12/SVM_Exercise_ML)  

---

## 📜 License
This project is licensed under the **MIT License** – free to use and share with credit.  

---

✨ *Star this repository if you found it useful for learning SVM!*  
