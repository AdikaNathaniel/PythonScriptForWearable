import time
import random
from datetime import datetime
# import requests  # Commented out since we're not sending data now

# ML_MODEL_URL = "http://localhost:5000/predict"  # Commented out for now

def generate_fake_vitals():
    return {
        "systolic_bp": round(random.uniform(100, 140), 1),    # mmHg
        "diastolic_bp": round(random.uniform(60, 90), 1),      # mmHg
        "temperature": round(random.uniform(36.5, 37.5), 1),   # °C
        "heart_rate": round(random.uniform(60, 100), 0),       # bpm
        "oxygen_sat": round(random.uniform(95, 100), 1)        # %
    }

while True:
    # 1. Generate vitals
    vitals = generate_fake_vitals()

    # 2. Format input for ML model (as list if needed later)
    input_data = [
        vitals["systolic_bp"],
        vitals["diastolic_bp"],
        vitals["temperature"],
        vitals["heart_rate"],
        vitals["oxygen_sat"]
    ]

    # 3. Display each value clearly
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n[{timestamp}] Sending the following vitals (simulation):")
    print(f"  ➤ Systolic Blood Pressure   : {vitals['systolic_bp']} mmHg")
    print(f"  ➤ Diastolic Blood Pressure  : {vitals['diastolic_bp']} mmHg")
    print(f"  ➤ Body Temperature          : {vitals['temperature']} °C")
    print(f"  ➤ Heart Rate                : {vitals['heart_rate']} bpm")
    print(f"  ➤ Oxygen Saturation         : {vitals['oxygen_sat']} %")
    print("  → Prediction: (pending...)\n")

    # 4. Wait for 30 seconds
    time.sleep(30)
