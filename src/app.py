from fastapi import FastAPI
import joblib
import pandas as pd
import time

app = FastAPI()
model = joblib.load("models/xgb_income_model.joblib")

@app.post("/predict")
def predict(features: dict):
    start = time.time()
    df = pd.DataFrame([features])
    prediction = model.predict(df)[0]
    latency = time.time() - start
    return {"prediction": int(prediction), "latency_seconds": latency}