import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Online Shopper Purchase Prediction",
    layout="wide"
)

st.title("Online Shopper Purchase Intention Prediction")

st.write(
    """
    This application predicts whether an online shopping session
    is likely to result in a purchase.

    Upload the provided test_data.csv file and select a machine
    learning model to view its performance.
    """
)

# --------------------------------------------------
# Load trained models
# --------------------------------------------------

models = {
    "Logistic Regression":
        joblib.load("model/logistic_regression.pkl"),

    "Decision Tree":
        joblib.load("model/decision_tree.pkl"),

    "K-Nearest Neighbors":
        joblib.load("model/knn.pkl"),

    "Naive Bayes":
        joblib.load("model/naive_bayes.pkl"),

    "Random Forest":
        joblib.load("model/random_forest.pkl")
}

scaler = joblib.load("model/scaler.pkl")


# --------------------------------------------------
# File upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Test Data CSV",
    type=["csv"]
)


if uploaded_file is not None:

    test_df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Test Data")
    st.dataframe(test_df.head(10))

    # Check Revenue column
    if "Revenue" not in test_df.columns:

        st.error(
            "The uploaded CSV must contain the Revenue column."
        )

    else:

        # --------------------------------------------------
        # Prepare test data
        # --------------------------------------------------

        X_test = test_df.drop("Revenue", axis=1)
        y_test = test_df["Revenue"].astype(int)

        # Encode categorical variables
        X_test = pd.get_dummies(
            X_test,
            columns=["Month", "VisitorType"],
            drop_first=True
        )

        X_test["Weekend"] = X_test["Weekend"].astype(int)

        # These columns must match the training dataset
        expected_columns = [
            'Administrative',
            'Administrative_Duration',
            'Informational',
            'Informational_Duration',
            'ProductRelated',
            'ProductRelated_Duration',
            'BounceRates',
            'ExitRates',
            'PageValues',
            'SpecialDay',
            'OperatingSystems',
            'Browser',
            'Region',
            'TrafficType',
            'Weekend',
            'Month_Dec',
            'Month_Feb',
            'Month_Jul',
            'Month_June',
            'Month_Mar',
            'Month_May',
            'Month_Nov',
            'Month_Oct',
            'Month_Sep',
            'VisitorType_Other',
            'VisitorType_Returning_Visitor'
        ]

        # Add any missing dummy columns
        for col in expected_columns:
            if col not in X_test.columns:
                X_test[col] = 0

        # Keep columns in correct order
        X_test = X_test[expected_columns]


        # --------------------------------------------------
        # Model selection
        # --------------------------------------------------

        selected_model_name = st.selectbox(
            "Select Machine Learning Model",
            list(models.keys())
        )

        selected_model = models[selected_model_name]


        # --------------------------------------------------
        # Prediction
        # --------------------------------------------------

        if selected_model_name in [
            "Logistic Regression",
            "K-Nearest Neighbors"
        ]:

            X_input = scaler.transform(X_test)

        else:

            X_input = X_test


        y_pred = selected_model.predict(X_input)

        y_prob = selected_model.predict_proba(X_input)[:, 1]


        # --------------------------------------------------
        # Evaluation metrics
        # --------------------------------------------------

        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        mcc = matthews_corrcoef(
            y_test,
            y_pred
        )


        st.subheader(
            f"Performance: {selected_model_name}"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Accuracy",
            f"{accuracy:.4f}"
        )

        col2.metric(
            "AUC",
            f"{auc:.4f}"
        )

        col3.metric(
            "Precision",
            f"{precision:.4f}"
        )


        col4, col5, col6 = st.columns(3)

        col4.metric(
            "Recall",
            f"{recall:.4f}"
        )

        col5.metric(
            "F1 Score",
            f"{f1:.4f}"
        )

        col6.metric(
            "MCC",
            f"{mcc:.4f}"
        )


        # --------------------------------------------------
        # Confusion Matrix
        # --------------------------------------------------

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(
            y_test,
            y_pred
        )

        cm_df = pd.DataFrame(
            cm,
            index=[
                "Actual No Purchase",
                "Actual Purchase"
            ],
            columns=[
                "Predicted No Purchase",
                "Predicted Purchase"
            ]
        )

        st.dataframe(cm_df)


        # --------------------------------------------------
        # Classification Report
        # --------------------------------------------------

        st.subheader("Classification Report")

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(
            report
        ).transpose()

        st.dataframe(report_df)


        # --------------------------------------------------
        # Predictions
        # --------------------------------------------------

        st.subheader("Sample Predictions")

        prediction_df = test_df.copy()

        prediction_df["Predicted_Revenue"] = (
            y_pred.astype(bool)
        )

        st.dataframe(
            prediction_df.head(20)
        )