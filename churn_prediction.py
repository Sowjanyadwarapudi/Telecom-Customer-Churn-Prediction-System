
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the dataset
df = pd.read_csv("churn.csv")

# Convert TotalCharges to numeric values
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Select features and target
features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "InternetService"
]

X = df[features]
y = df["Churn"]

# Separate numerical and categorical columns
num = ["tenure", "MonthlyCharges", "TotalCharges"]
cat = ["Contract", "InternetService"]

# Preprocessing
pre = ColumnTransformer([
    (
        "num",
        SimpleImputer(strategy="median"),
        num
    ),
    (
        "cat",
        Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore"))
        ]),
        cat
    )
])

# Create the machine learning model
model = Pipeline([
    ("prep", pre),
    ("clf", RandomForestClassifier(random_state=42))
])

# Split the dataset into training and testing sets
Xtr, Xte, ytr, yte = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train the model
model.fit(Xtr, ytr)

# Evaluate the model
predictions = model.predict(Xte)

print("Classification Report:")
print(classification_report(yte, predictions))

# Get input from the user
print("\nEnter Customer Details:")

sample = pd.DataFrame([{
    "tenure": float(input("Tenure months: ")),
    "MonthlyCharges": float(input("Monthly charges: ")),
    "TotalCharges": float(input("Total charges: ")),
    "Contract": input("Contract type: "),
    "InternetService": input("Internet service: ")
}])

# Predict customer churn
result = model.predict(sample)[0]

print("\nPredicted Churn:", result)
