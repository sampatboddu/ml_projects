# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
from typing import List

app = FastAPI(title="Vertex AI Custom Prediction")

with open("/model/model.pkl", "rb") as f:
    model = pickle.load(f)

class PredictRequest(BaseModel):
    instances: List[List[float]]

class PredictResponse(BaseModel):
    predictions: List[int]
    probabilities: List[List[float]]

@app.post("/predict", response_model=PredictResponse)
async def predict(body: PredictRequest):
    try:
        X = np.array(body.instances, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("Expected 2D array")

        proba = model.predict_proba(X).tolist()
        preds = model.predict(X).tolist()

        return {"predictions": preds, "probabilities": proba}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}
