import logging, json, os, asyncio, time, httpx
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import numpy as np

# ------------------------------
# 🧱 INITIAL SETUP
# ------------------------------
app = FastAPI(title="🧠 MediLearn Controller (FedAvg + Async + Configurable)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

LOG_FILE = "federated.log"
STATUS_FILE = "latest_status.json"
HISTORY_FILE = "training_history.json"
GLOBAL_MODEL_FILE = "global_model.json"
CONFIG_FILE = "agent_config.json"
AGENT_LOG = "agent_log.json"

# Default configuration
CYCLE_COUNT = 3
HOSPITALS = [
    "http://127.0.0.1:8001/train",
    "http://127.0.0.1:8002/train",
    "http://127.0.0.1:8003/train"
]

# Logging setup
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logging.info("=== MediLearn Controller Initialized (FedAvg + Async + Configurable) ===")

# ------------------------------
# ⚙️ CONFIG UTILITIES
# ------------------------------
def load_config():
    global CYCLE_COUNT, HOSPITALS
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            cfg = json.load(f)
        CYCLE_COUNT = cfg.get("cycles", 3)
        HOSPITALS[:] = cfg.get("hospitals", HOSPITALS)
        logging.info(f"Loaded dynamic config → cycles={CYCLE_COUNT}, hospitals={len(HOSPITALS)}")

def save_status(data):
    with open(STATUS_FILE, "w") as f:
        json.dump(data, f, indent=2)

def log_history(entry):
    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)
        except json.JSONDecodeError:
            history = []
    history.append(entry)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

def log_agent_action(cycle, action):
    log = []
    if os.path.exists(AGENT_LOG):
        try:
            with open(AGENT_LOG, "r") as f:
                log = json.load(f)
        except json.JSONDecodeError:
            log = []
    log.append({"cycle": cycle, "action": action, "time": datetime.now().strftime("%H:%M:%S")})
    with open(AGENT_LOG, "w") as f:
        json.dump(log, f, indent=2)

# ------------------------------
# 🧠 FEDERATED AVERAGING HELPERS
# ------------------------------
def aggregate_fedavg(results):
    total_samples = sum(r.get("samples", 0) for r in results if r)
    if total_samples == 0:
        return 0.0
    weighted_sum = sum(r["accuracy"] * r["samples"] for r in results if r)
    return round(weighted_sum / total_samples, 3)

def aggregate_model_weights(results):
    valid = [r for r in results if "weights" in r]
    if not valid:
        return None
    coefs = [np.array(r["weights"][0]) for r in valid]
    intercepts = [np.array(r["weights"][1]) for r in valid]
    avg_coef = np.mean(coefs, axis=0).tolist()
    avg_intercept = np.mean(intercepts, axis=0).tolist()
    return [avg_coef, avg_intercept]

def save_global_model(weights):
    if weights:
        with open(GLOBAL_MODEL_FILE, "w") as f:
            json.dump(weights, f)
        logging.info("✅ Global model weights updated and saved.")

def load_global_model():
    if os.path.exists(GLOBAL_MODEL_FILE):
        with open(GLOBAL_MODEL_FILE, "r") as f:
            return json.load(f)
    return None

# ------------------------------
# ⚙️ ASYNC TRAINING
# ------------------------------
async def train_all_hospitals(global_weights):
    async with httpx.AsyncClient() as client:
        tasks = [client.post(url, json={"global_weights": global_weights}, timeout=15) for url in HOSPITALS]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    for i, res in enumerate(responses):
        if isinstance(res, Exception):
            results.append({"hospital": f"Hospital_{chr(65+i)}", "error": str(res)})
        else:
            results.append(res.json())
    return results

# ------------------------------
# 🔁 MAIN SIMULATION LOOP
# ------------------------------
def simulate_agent_cycle():
    load_config()
    global_weights = load_global_model()
    print("🧠 MediLearn Agent Simulation Started (Async + Global Weights)")
    logging.info("Simulation started by /start")

    global_data = {"hospitals": [], "global_accuracy": 0.0, "cycle": 0}

    for cycle in range(1, CYCLE_COUNT + 1):
        print(f"\n🚀 Cycle {cycle} started...")
        log_agent_action(cycle, f"Starting cycle {cycle}")

        results = asyncio.run(train_all_hospitals(global_weights))

        global_accuracy = aggregate_fedavg(results)
        new_global_weights = aggregate_model_weights(results)
        save_global_model(new_global_weights)

        global_data.update({
            "global_accuracy": global_accuracy,
            "hospitals": results,
            "cycle": cycle,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        save_status(global_data)
        log_history(global_data)

        logging.info(f"Cycle {cycle} → Global Accuracy: {global_accuracy}")
        log_agent_action(cycle, f"Completed cycle {cycle} with acc={global_accuracy}")

        global_weights = new_global_weights
        time.sleep(1)

    print("✅ Simulation completed successfully!")
    logging.info("Simulation completed successfully.")

# ------------------------------
# 🌐 API ROUTES
# ------------------------------
@app.post("/start")
def start_simulation(background_tasks: BackgroundTasks):
    background_tasks.add_task(simulate_agent_cycle)
    return {"message": "🚀 MediLearn Agent simulation started."}

@app.post("/config")
def update_config(cfg: dict):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)
    load_config()
    return {"message": "✅ Config updated.", "config": cfg}

@app.get("/status")
def get_status():
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE, "r") as f:
            return json.load(f)
    raise HTTPException(404, "No training data yet. Run /start first.")

@app.get("/history")
def get_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

@app.get("/global_model")
def get_global_model():
    if os.path.exists(GLOBAL_MODEL_FILE):
        with open(GLOBAL_MODEL_FILE, "r") as f:
            return json.load(f)
    return {"message": "No global weights yet."}

@app.get("/agent_log")
def get_agent_log():
    if os.path.exists(AGENT_LOG):
        with open(AGENT_LOG, "r") as f:
            return json.load(f)
    return []

@app.get("/stream")
async def stream_status():
    async def event_stream():
        while True:
            if os.path.exists(STATUS_FILE):
                with open(STATUS_FILE, "r") as f:
                    yield f"data: {f.read()}\n\n"
            await asyncio.sleep(2)
    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/reset")
def reset():
    for file in [STATUS_FILE, HISTORY_FILE, GLOBAL_MODEL_FILE, CONFIG_FILE, AGENT_LOG]:
        if os.path.exists(file):
            os.remove(file)
    logging.info("Simulation reset.")
    return {"message": "🧹 Simulation reset successfully."}

@app.get("/health")
def health_check():
    return {"status": "🧠 MediLearn Controller Active ✅", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

@app.get("/")
def home():
    return {"status": "MediLearn Controller 🩺", "version": "4.0 (Async + Configurable + FedAvg)"}
