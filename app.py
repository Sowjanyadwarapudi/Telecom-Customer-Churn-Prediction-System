
import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Page configuration
st.set_page_config(
    page_title="Telecom Customer Churn Prediction",
    page_icon="📱",
    layout="centered"
)

# Application heading
st.title("📱 Telecom Customer Churn Prediction")
st.write(
    "Predict whether a telecom customer is likely "
    "to leave the service."
)

# Load and train the model
@st.cache_resource
def train_model():
    df = pd.read_csv("churn.csv")

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

    features = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Contract",
        "InternetService"
    ]

    X = df[features]
    y = df["Churn"]

    num = ["tenure", "MonthlyCharges", "TotalCharges"]
    cat = ["Contract", "InternetService"]

    pre = ColumnTransformer([
        (
            "num",
            SimpleImputer(strategy="median"),
            num
        ),
        (
            "cat",
            Pipeline([
                (
                    "impute",
                    SimpleImputer(strategy="most_frequent")
                ),
                (
                    "ohe",
                    OneHotEncoder(handle_unknown="ignore")
                )
            ]),
            cat
        )
    ])

    model = Pipeline([
        ("prep", pre),
        ("clf", RandomForestClassifier(random_state=42))
    ])

    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model.fit(Xtr, ytr)

    predictions = model.predict(Xte)
    accuracy = accuracy_score(yte, predictions)

    report = classification_report(
        yte,
        predictions,
        output_dict=True,
        zero_division=0
    )

    return model, accuracy, report


# Handle loading errors
try:
    model, accuracy, report = train_model()
except Exception as e:
    st.error(
        "Could not load the dataset or train the model. "
        "Check that churn.csv is in the project folder "
        "and contains the required columns."
    )
    st.exception(e)
    st.stop()

# Customer details form
st.subheader("👤 Enter Customer Service Details")

with st.form("customer_form"):

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12
    )

    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        value=70.5,
        step=1.0
    )

    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        value=850.0,
        step=10.0
    )

    contract = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    submitted = st.form_submit_button(
        "🔍 Predict Customer Churn",
        use_container_width=True
    )


# Make prediction
if submitted:

    sample = pd.DataFrame([{
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Contract": contract,
        "InternetService": internet_service
    }])

    prediction = model.predict(sample)[0]

    st.subheader("Prediction Result")

    if prediction == "Yes":
        st.error("Predicted Churn: YES")
    else:
        st.success("Predicted Churn: NO")


st.divider()
