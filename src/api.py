from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="Real-Time Fraud Detection API")

# Load the trained model once, at startup
model = joblib.load("../models/final_model.pkl")

# Define exactly what a request must look like
class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}

@app.post("/predict")
def predict(transaction: Transaction):
    # Convert the incoming request into a DataFrame matching training column order
    data = pd.DataFrame([transaction.dict()])

    fraud_probability = model.predict_proba(data)[0][1]
    prediction = int(fraud_probability >= 0.5)

    return {
        "fraud_prediction": prediction,
        "fraud_probability": round(float(fraud_probability), 4),
        "risk_level": "HIGH" if fraud_probability >= 0.7 else "MEDIUM" if fraud_probability >= 0.3 else "LOW"
    }