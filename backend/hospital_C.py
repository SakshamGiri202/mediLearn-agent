from fastapi import FastAPI, Request
from datetime import datetime
import sys, os

# Add ml_core to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_core.train_local import train_on_local_data

app = FastAPI(title="🏥 Hospital C API (Federated Node)")

DATASET_NAME = "stroke.csv"  # Local dataset for Hospital C


@app.post("/train")
async def train_model(request: Request):
    """Local training endpoint — accepts optional global weights."""
    body = await request.json()
    global_weights = body.get("global_weights")

    local_weights, accuracy, samples = train_on_local_data(DATASET_NAME, global_weights)

    result = {
        "hospital": "Hospital_C",
        "accuracy": round(float(accuracy), 3),
        "samples": int(samples),
        "weights": local_weights,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    return result


@app.get("/")
def home():
    return {"status": "Hospital C Node active ✅"}
