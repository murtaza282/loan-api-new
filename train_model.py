"""
Train and save the model.
==========================
Run this ONCE to create loan_pipeline.joblib, which the API loads.
This is the same packaged pipeline from the Session 4 notebook.

Run with:   python train_model.py
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Load and clean
df = pd.read_csv("loan_data.csv").drop(columns=["customer_id"])
for col in ["annual_income", "credit_score", "years_employed"]:
    df[col] = df[col].fillna(df[col].median())

X = df.drop(columns=["defaulted"])
y = df["defaulted"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build and train the pipeline (scaler + model in one object)
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000)),
])
pipeline.fit(X_train, y_train)

# Save it - this one file is what the API serves
joblib.dump(pipeline, "loan_pipeline.joblib")

print("Saved loan_pipeline.joblib")
print("Test accuracy:", round(pipeline.score(X_test, y_test), 3))
