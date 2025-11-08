import streamlit as st
import pandas as pd
import requests
import time  # NEW: We need this to make the dashboard auto-refresh

# --- Page Setup ---
st.set_page_config(
    page_title="MediLearn Agent Dashboard",
    page_icon="🧠",
    layout="wide"
)

# --- Backend URL ---
# This is the address of your backend server
BACKEND_URL = "http://127.0.0.1:8000"

# --- Session State for Chart ---
# NEW: We initialize "session state" to store the chart's history.
# This acts like a "memory" for your app.
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["Cycle", "Global Accuracy"])

# --- Title & Description ---
st.title("🧠 MediLearn Agent Dashboard")
st.write("Live dashboard showing the federated learning agent's progress.")

# --- Start Button ---
# This part stays the same
if st.button("🚀 Start New Training Round"):
    try:
        response = requests.post(f"{BACKEND_URL}/start")
        if response.status_code == 200:
            st.success("Started new training round! Fetching results...")
            # NEW: Clear the chart history for a new round
            st.session_state.history = pd.DataFrame(columns=["Cycle", "Global Accuracy"])
        else:
            st.error(f"Error starting round: {response.text}")
    except requests.exceptions.ConnectionError:
        st.error(f"Failed to connect to backend. Is it running at {BACKEND_URL}?")

st.divider()

# --- NEW: Live Data Fetching Logic ---
try:
    # We try to get the data from the /status endpoint
    status_response = requests.get(f"{BACKEND_URL}/status")
    
    if status_response.status_code == 200:
        status_data = status_response.json()

        # Parse the JSON data (based on your screenshot)
        global_acc = status_data.get('global_accuracy', 0)
        cycle = status_data.get('cycle', 0)
        timestamp = status_data.get('timestamp', 'N/A')
        hospitals = status_data.get('hospitals', [])

        # Process hospital data into a simple dictionary
        hospital_data = {}
        for h in hospitals:
            hospital_data[h.get('hospital')] = h.get('accuracy', 0)

        # --- Main Metrics (NOW WITH LIVE DATA) ---
        st.subheader("Current Model Accuracy")
        
        # We need 5 columns to fit everything
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # Replace all the "fake" values with our new variables
        col1.metric(label="🔁 Cycle", value=cycle)
        col2.metric(label="🧠 Global Accuracy", value=f"{global_acc * 100:.1f}%")
        col3.metric(label="🏥 Hospital A", value=f"{hospital_data.get('Hospital_A', 0) * 100:.1f}%")
        col4.metric(label="🏥 Hospital B", value=f"{hospital_data.get('Hospital_B', 0) * 100:.1f}%")
        col5.metric(label="🏥 Hospital C", value=f"{hospital_data.get('Hospital_C', 0) * 100:.1f}%")

        # --- Agent Status (NOW WITH LIVE DATA) ---
        st.subheader("Agent Status")
        st.info(f"STATUS: Data last updated at {timestamp}")

        # --- Accuracy Chart (NOW WITH LIVE DATA) ---
        st.subheader("Global Accuracy Over Time")

        # Check if the cycle is new and not already in our history
        if cycle > 0 and (st.session_state.history.empty or st.session_state.history.iloc[-1]["Cycle"] != cycle):
            new_row = pd.DataFrame({"Cycle": [cycle], "Global Accuracy": [global_acc]})
            st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True)
        
        # Plot the REAL data from our session state "memory"
        if not st.session_state.history.empty:
            st.line_chart(st.session_state.history, x="Cycle", y="Global Accuracy")
        else:
            st.write("Waiting for first training cycle to complete...")

    else:
        st.error(f"Backend sent an error (Code {status_response.status_code}). Is it ready?")

except requests.exceptions.ConnectionError:
    st.error(f"Failed to connect to backend. Is the server running at {BACKEND_URL}?")
except requests.exceptions.JSONDecodeError:
    st.error("Backend sent invalid data. It might be restarting.")

# --- NEW: Auto-Refresh ---
# This is the final and most important part.
# We tell Streamlit to wait 3 seconds, then re-run the *entire script* from the top.
# This creates a live-updating dashboard.
time.sleep(3)
st.rerun()