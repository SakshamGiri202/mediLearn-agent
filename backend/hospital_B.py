from fastapi import FastAPI
from datetime import datetime
import random

app = FastAPI(title="Hospital B API (Simulated)")

@app.get("/train")
def train_model():
    """Simulated local training for Hospital B."""
    accuracy = round(random.uniform(0.82, 0.92), 3)
    weights = [[round(random.uniform(0.1, 0.9), 2) for _ in range(3)]]
    return {
        "hospital": "Hospital_B",
        "accuracy": accuracy,
        "weights": weights,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
