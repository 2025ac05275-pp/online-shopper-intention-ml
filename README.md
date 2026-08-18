# Online Shoppers Purchasing Intention Prediction

## 1. Problem Statement

The objective of this project is to develop and compare multiple machine learning classification models to predict whether an online shopping session will result in a purchase.

The target variable is **Revenue**:

* `True` means the shopping session resulted in a purchase.
* `False` means the shopping session did not result in a purchase.

The models are evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

---

## 2. Dataset Description

The **Online Shoppers Purchasing Intention Dataset** is used for this project.

The dataset contains information about online shopping sessions, including visitor behavior, page interaction, traffic information, and session-related attributes.

### Dataset Summary

* Number of instances: **12,330**
* Number of original input features: **17**
* Target variable: **Revenue**
* Classification type: **Binary Classification**
* Positive class: Purchase completed
* Negative class: Purchase not completed

Some of the important features include:

* Administrative
* Administrative Duration
* Informational
* Informational Duration
* Product Related
* Product Related Duration
* Bounce Rates
* Exit Rates
* Page Values
* Special Day
* Month
* Operating Systems
* Browser
* Region
* Traffic Type
* Visitor Type
* Weekend

The dataset is imbalanced, as the number of non-purchase sessions is significantly higher than the number of purchase sessions.

Target distribution observed in the dataset:

* No Purchase: **10,422**
* Purchase: **1,908**

Because of this imbalance, metrics such as Recall, F1 Score, AUC, and MCC are also considered along with Accuracy.

---

## 3. GitHub Repository Link

GitHub Repository:
https://github.com/2025ac05275-pp/online-shopper-intention-ml

---

## 4. Machine Learning Models Used

The following classification models were implemented on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Random Forest Classifier

The dataset was divided into training and testing sets using an 80:20 split.

Stratified sampling was used to maintain approximately the same Revenue class distribution in both training and testing data.

Feature scaling using `StandardScaler` was applied for Logistic Regression and K-Nearest Neighbors.

Categorical features such as Month and Visitor Type were converted into numerical features using one-hot encoding.

---

## 5. Model Performance Comparison

| ML Model Name       |   Accuracy |        AUC | Precision | Recall |   F1 Score |        MCC |
| ------------------- | ---------: | ---------: | --------: | -----: | ---------: | ---------: |
| Logistic Regression |     0.8812 |     0.8873 |    0.7432 | 0.3560 |     0.4814 |     0.4603 |
| Decision Tree       |     0.8646 |     0.7360 |    0.5645 | 0.5497 |     0.5570 |     0.4772 |
| K-Nearest Neighbors |     0.8686 |     0.7725 |    0.6368 | 0.3534 |     0.4545 |     0.4085 |
| Naive Bayes         |     0.7960 |     0.8095 |    0.3951 | 0.5969 |     0.4755 |     0.3670 |
| Random Forest       | **0.8978** | **0.9179** |    0.7321 | 0.5366 | **0.6193** | **0.5710** |

---

## 6. Model Performance Observations

### Logistic Regression

Logistic Regression achieved good Accuracy and AUC. It also showed strong Precision, which indicates that many of the purchase predictions made by the model were correct.

Its Recall was relatively low, which means that it failed to identify a considerable number of actual purchase sessions.

### Decision Tree

The Decision Tree achieved slightly lower Accuracy and AUC compared with Logistic Regression and Random Forest.

Its Recall was better than Logistic Regression and KNN, which means it was able to identify more of the actual purchase cases.

### K-Nearest Neighbors

KNN achieved reasonable Accuracy but showed comparatively low Recall and F1 Score.

This indicates that the model performed better on the majority non-purchase class but was less effective at identifying purchase sessions.

### Naive Bayes

Naive Bayes achieved the highest Recall among the tested models.

This means it identified a relatively high proportion of actual purchases. However, its Precision was low, which indicates that it also produced a larger number of false positive purchase predictions.

### Random Forest

Random Forest provided the strongest overall performance.

It achieved the highest:

* Accuracy
* AUC
* F1 Score
* MCC

It also provided a better balance between Precision and Recall compared with most of the other models.

### Overall Winner

**Random Forest is selected as the overall best-performing model for this dataset.**

It achieved the strongest overall combination of Accuracy, AUC, F1 Score, and MCC.

---

## 7. Streamlit Application

An interactive Streamlit web application was created to demonstrate the trained machine learning models.

The application allows the user to:

* Upload test data in CSV format
* Select a machine learning model
* View model evaluation metrics
* View the confusion matrix
* View the classification report
* View sample predictions

### Live Streamlit Application

`TO_BE_ADDED_AFTER_DEPLOYMENT`

---

## 8. Project Structure

```text
ML_Assignment_2/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── online_shoppers_intention.csv
├── online_shopper_models.ipynb
│
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    └── scaler.pkl
```

---

## 9. Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Joblib
* Jupyter Notebook

---

## 10. Conclusion

This project demonstrates an end-to-end machine learning classification workflow starting from dataset preprocessing and model training to evaluation and deployment.

Among the evaluated models, Random Forest produced the best overall performance for predicting online shopper purchase intention.
