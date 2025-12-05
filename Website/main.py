from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import math
import random
from inference import InferenceModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model
try:
    model = InferenceModel()
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

class SimulationInput(BaseModel):
    lamda: List[float]
    r: List[float]

@app.get("/")
def read_root():
    return {"message": "Scientific Backend is Running"}

@app.post("/calculate")
def calculate_simulation(data: SimulationInput):
    if len(data.lamda) != len(data.r):
        raise HTTPException(status_code=400, detail="Lambda and R lists must be the same length")

    for val in data.r:
        if not (0 <= val <= 1):
            raise HTTPException(status_code=400, detail="R values must be between 0 and 1")
    
    if model is None:
         raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        final_data = model.predict(data.lamda, data.r)
        return {"data": final_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)