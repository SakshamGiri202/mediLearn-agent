"""
Hospital Manager Service
-------------------------
Handles:
✅ Dynamic hospital creation (auto-launch on available port)
✅ Optional dataset upload (CSV or preset)
✅ Registration in agent_config.json
✅ Human-readable logging
✅ Graceful deletion and listing of hospital nodes
"""

from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os, subprocess, json, logging, shutil

# ---------------------------------------------------
# ⚙️ CONFIG
# ---------------------------------------------------
CONFIG_FILE = "agent_config.json"
HOSPITALS_DIR = "backend"
DATASET_DIR = "ml_core/dataset"

app = FastAPI(title="🏥 MediLearn Hospital Manager")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------
# 🧩 JSON Helpers
# ---------------------------------------------------
def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"hospitals": [], "cycles": 3}


def save_config(data):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------
# ➕ ADD NEW HOSPITAL
# ---------------------------------------------------
@app.post("/add_hospital")
async def add_hospital(
    hospital_name: str = Form(...),
    dataset_name: str = Form(...),
    port: str = Form(...),
    autostart: str = Form("true"),
    file: UploadFile | None = None,
):
    hospital_name = hospital_name.strip().replace(" ", "_")
    script_path = os.path.join(HOSPITALS_DIR, f"hospital_{hospital_name}.py")
    dataset_path = os.path.join(DATASET_DIR, dataset_name)

    # 📦 Handle custom dataset upload
    if file:
        uploaded_path = os.path.join(DATASET_DIR, file.filename)
        with open(uploaded_path, "wb") as f:
            f.write(await file.read())
        dataset_path = uploaded_path
        dataset_name = file.filename

    # 🧠 Create hospital script dynamically
    hospital_script = f"""
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from ml_core.train_local import train_on_local_data

app = FastAPI(title="{hospital_name}")
HOSPITAL_NAME = "{hospital_name}"

@app.post("/train")
async def train(request: Request):
    payload = await request.json()
    global_weights = payload.get("global_weights")
    weights, acc, samples, features = train_on_local_data("{dataset_name}", global_weights)
    return JSONResponse({{
        "weights": weights,
        "accuracy": acc,
        "samples": samples,
        "hospital": HOSPITAL_NAME
    }})

@app.get("/health")
async def health():
    return {{"status": "running", "hospital": HOSPITAL_NAME}}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port={port})
"""
    os.makedirs(HOSPITALS_DIR, exist_ok=True)
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(hospital_script)

    # 🗂 Update Config
    cfg = load_config()
    hospitals = cfg.get("hospitals", [])

    new_entry = {
        "name": hospital_name,
        "port": int(port),
        "endpoint": f"http://127.0.0.1:{port}/train"
    }

    # Avoid duplicates
    if not any(h.get("endpoint") == new_entry["endpoint"] for h in hospitals if isinstance(h, dict)):
        hospitals.append(new_entry)

    cfg["hospitals"] = hospitals
    save_config(cfg)

    logging.info(f"🏥 Added new hospital → {hospital_name} (Port {port})")
    print(f"✅ Registered: {hospital_name} → http://127.0.0.1:{port}/train")

    # 🚀 Autostart if enabled
    if autostart.lower() == "true":
        subprocess.Popen(["python", script_path])
        logging.info(f"🚀 Auto-started hospital {hospital_name} on port {port}")

    return {
        "message": f"✅ {hospital_name} created successfully",
        "hospital_name": hospital_name,
        "dataset": dataset_name,
        "port": port,
        "endpoint": f"http://127.0.0.1:{port}/train",
    }


# ---------------------------------------------------
# 🗑 REMOVE HOSPITAL
# ---------------------------------------------------
@app.post("/remove_hospital")
async def remove_hospital(hospital_name: str = Form(...)):
    hospital_name = hospital_name.strip().replace(" ", "_")
    script_path = os.path.join(HOSPITALS_DIR, f"hospital_{hospital_name}.py")

    cfg = load_config()
    hospitals = cfg.get("hospitals", [])
    new_hospitals = [h for h in hospitals if h.get("name") != hospital_name]
    cfg["hospitals"] = new_hospitals
    save_config(cfg)

    if os.path.exists(script_path):
        os.remove(script_path)
        deleted = True
        logging.info(f"🗑 Removed hospital script: {script_path}")
    else:
        deleted = False
        logging.warning(f"⚠️ No script found for {hospital_name}")

    return {"message": f"Hospital {hospital_name} removed", "script_deleted": deleted}


# ---------------------------------------------------
# 📋 LIST HOSPITALS
# ---------------------------------------------------
@app.get("/list_hospitals")
def list_hospitals():
    cfg = load_config()
    hospitals = cfg.get("hospitals", [])

    hospital_names = []
    for h in hospitals:
        if isinstance(h, dict):  # ✅ new structured format
            hospital_names.append(h.get("name", "Unknown"))
        elif isinstance(h, str):  # 🕰️ backward compatibility for old format
            # Extract port and guess name
            port = h.split(":")[-1].split("/")[0]
            hospital_names.append(f"Hospital_{port}")
        else:
            hospital_names.append(str(h))

    return {"registered_hospitals": hospital_names}


# ---------------------------------------------------
# 🩺 HEALTH CHECK
# ---------------------------------------------------
@app.get("/health")
def health():
    return {"status": "Hospital Manager running ✅"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8600)
