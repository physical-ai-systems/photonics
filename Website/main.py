from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import math
import random

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    
    unsorted_layers = []
    
    for i, (lam, r_val) in enumerate(zip(data.lamda, data.r)):
        thickness = round(random.uniform(0.5, 2.0), 2)
        random_color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        
        layer = {
            "name": f"Meta Layer {i+1}",
            "lamda": lam,      
            "r_val": r_val,
            "color": random_color,
            "thickness": thickness,
            "val": round(random.random(), 3),
            "ref_index": round(random.uniform(1.4, 3.5), 3),
            "impedance": round(random.uniform(300, 400), 2)
        }
        unsorted_layers.append(layer)


    sorted_layers = sorted(unsorted_layers, key=lambda x: x['lamda'])
    current_z_position = 0.0
    final_data = []

    for layer in sorted_layers:
        layer['start_z'] = current_z_position
        final_data.append(layer)
        
        current_z_position += layer['thickness']

    return {"data": final_data}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)