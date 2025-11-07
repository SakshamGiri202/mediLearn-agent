from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import requests, time, json, os, random
from datetime import datetime

app = FastAPI(title="MediLearn Controller (Simulated)")

# Allow dashboard or other tools to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATUS_FILE = "latest_status.json"

HOSPITALS = [
    "http://127.0.0.1:8001/train",
    "http://127.0.0.1:8002/train",
    "http://127.0.0.1:8003/train"
]

def simulate_agent_cycle():
    """Simulates MediLearn Agent visiting hospitals and aggregating results."""
    print("🧠 MediLearn Agent Simulation Started")

    global_data = {"hospitals": [], "global_accuracy": 0.0, "cycle": 0}
    for cycle in range(1, 4):
        results = []
        print(f"\n🚀 Cycle {cycle} started...")
        for url in HOSPITALS:
            try:
                response = requests.get(url)
                data = response.json()
                results.append(data)
                print(f"🏥 Visited {data['hospital']} → Accuracy: {data['accuracy']}")
            except Exception as e:
                print(f"❌ Error contacting {url}: {e}")

        # Simulate global aggregation
        global_accuracy = round(sum(r["accuracy"] for r in results) / len(results), 3)
        global_data["global_accuracy"] = global_accuracy
        global_data["hospitals"] = results
        global_data["cycle"] = cycle
        global_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Save progress for dashboard
        with open(STATUS_FILE, "w") as f:
            json.dump(global_data, f, indent=2)

        print(f"✅ Global accuracy after cycle {cycle}: {global_accuracy}\n")
        time.sleep(2)  # simulate training delay

    print("🧠 Simulation completed successfully!")


@app.post("/start")
def start_training(background_tasks: BackgroundTasks):
    """Start the MediLearn Agent simulation."""
    background_tasks.add_task(simulate_agent_cycle)
    return {"message": "MediLearn Agent simulation started."}


@app.get("/status")
def get_status():
    """Return latest simulation status."""
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE, "r") as f:
            return json.load(f)
    return {"message": "No training data yet. Run /start to begin simulation."}


@app.get("/")
def home():
    return {"status": "MediLearn Controller active ✅"}
