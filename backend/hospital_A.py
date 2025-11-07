import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from datetime import datetime
import random

app = FastAPI(title="Hospital A API (Simulated)")

@app.get("/train")
def train_model():
    """Simulated local training for Hospital A."""
    accuracy = round(random.uniform(0.80, 0.90), 3)
    weights = [[round(random.uniform(0.1, 0.9), 2) for _ in range(3)]]
    return {
        "hospital": "Hospital_A",
        "accuracy": accuracy,
        "weights": weights,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
