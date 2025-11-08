from fastapi import FastAPI, Request
from datetime import datetime
import sys, os

# Add ml_core to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_core.train_local import train_on_local_data

app = FastAPI(title="🏥 Hospital A API (Federated Node)")

DATASET_NAME = "heart_disease.csv"  # Local dataset for Hospital A


@app.post("/train")
async def train_model(request: Request):
    """Local training endpoint — accepts optional global weights."""
    body = await request.json()
    global_weights = body.get("global_weights")  # from controller, may be None first round

    local_weights, accuracy, samples = train_on_local_data(DATASET_NAME, global_weights)

    result = {
        "hospital": "Hospital_A",
        "accuracy": round(float(accuracy), 3),
        "samples": int(samples),
        "weights": local_weights,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    return result


@app.get("/")
def home():
    return {"status": "Hospital A Node active ✅"}
