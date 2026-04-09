from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import lightgbm as lgb
import joblib

app = FastAPI(title="Hackathon Model API", version="1.0")

# Load your trained model globally so it stays in memory
# Ensure you have a 'model.pkl' in your directory, or adjust to your model's format
try:
    model_point = joblib.load("model_point.pkl")
except Exception as e:
    model_point = None
    print("Warning: Model not found. Please ensure model_point.pkl exists.")

class PredictRequest(BaseModel):
    # Define the expected input features based on your dataset
    TempAmbient: float
    TempSetPoint: float
    YardVolume: float
    Hour: int
    DayOfWeek: int
    Month: int

@app.get("/")
def health_check():
    return {"status": "API is running"}

@app.post("/predict")
def predict(request: PredictRequest):
    if model_point is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    # Convert incoming JSON data to a DataFrame
    input_data = pd.DataFrame([request.model_dump() if hasattr(request, 'model_dump') else request.dict()])
    
    # Generate prediction
    prediction = model_point.predict(input_data)
    
    return {
        "predicted_power_load": float(prediction[0])
    }
