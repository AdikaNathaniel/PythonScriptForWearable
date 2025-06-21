# Updated clinical_monitoring.py
import time
import random
from datetime import datetime
import json
from kafka import KafkaProducer
import os
import pandas as pd
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Configuration
WORKING_DIR = r"C:\PregnancyMonitor\DataTransmission"
MODEL_PATH = "severity_model.pkl"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "severity_encoder.pkl"
INPUT_FILE = "input.txt"
PATIENT_ID = "001"
INTERVAL_SECONDS = 5

# Kafka Configuration
KAFKA_BROKERS = 'localhost:9092'
TOPIC_NAME = 'vitals-updates'

class KafkaVitalsProducer:
    def __init__(self):
        self.producer = KafkaProducer(
            bootstrap_servers=[KAFKA_BROKERS],
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
    
    def send_vitals(self, vitals_data):
        try:
            self.producer.send(TOPIC_NAME, value=vitals_data)
            print(f"📤 Sent vitals data to Kafka: {vitals_data}")
        except Exception as e:
            print(f"❌ Error sending vitals data: {e}")

# Initialize Kafka producer
vitals_producer = KafkaVitalsProducer()

def setup_environment():
    """Verify all files exist."""
    os.chdir(WORKING_DIR)
    print(f"✓ Working in: {WORKING_DIR}")
    
    if not os.path.exists("realistic_noisy_preeclampsia_dataset_noisy.csv"):
        raise FileNotFoundError("Missing CSV dataset file")

def calculate_severity(df):
    """Calculate severity levels based on clinical thresholds"""
    conditions = [
        (df['MAP'] >= 70) & (df['MAP'] <= 100) & (df['Protein_Urine'] == 0),
        (df['MAP'] >= 107) & (df['MAP'] <= 113) & (df['Protein_Urine'] >= 1),
        (df['MAP'] >= 114) & (df['MAP'] <= 129) & (df['Protein_Urine'] >= 2),
        (df['MAP'] >= 130) & (df['Protein_Urine'] >= 2),
        (df['MAP'] >= 107) & (df['MAP'] <= 113) & (df['Protein_Urine'] < 1),
        (df['MAP'] >= 114) & (df['MAP'] <= 129) & (df['Protein_Urine'] < 2),
    ]
    
    choices = [
        'Normal',
        'Mild',
        'Moderate',
        'Severe',
        'Mild',
        'Moderate'
    ]
    
    result = np.select(conditions, choices, default='Normal')
    normal_map_mask = (df['MAP'] >= 70) & (df['MAP'] <= 100)
    result = np.where(normal_map_mask, 'Normal', result)
    
    return result

def load_and_preprocess_data():
    """Load data and create severity labels"""
    df = pd.read_csv("realistic_noisy_preeclampsia_dataset_noisy.csv")
    df['MAP'] = df.get('MAP', (df['Systolic_BP'] + 2 * df['Diastolic_BP']) / 3)
    df['Severity'] = calculate_severity(df)
    encoder = LabelEncoder()
    df['Severity_Encoded'] = encoder.fit_transform(df['Severity'])
    features = ['Systolic_BP', 'Diastolic_BP', 'Protein_Urine', 'MAP']
    X = df[features]
    y = df['Severity_Encoded']
    return X, y, encoder, features

def train_and_save_model():
    """Train and save severity prediction model"""
    X, y, encoder, features = load_and_preprocess_data()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42)
    model.fit(X_scaled, y)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    print("✅ Severity model trained and saved")
    return features

def classify_manual(systolic, diastolic, protein, map_val):
    """Manual classification based on clinical guidelines"""
    if 70 <= map_val <= 100:
        return "Normal", "MAP in normal range (70-100)"
    elif 107 <= map_val <= 113:
        if protein >= 1:
            return "Mild", f"MAP: {map_val:.1f} (107-113), Protein: {protein}+ (≥1)"
        return "Mild", f"MAP: {map_val:.1f} (107-113), Protein: {protein} (below threshold)"
    elif 114 <= map_val <= 129:
        if protein >= 2:
            return "Moderate", f"MAP: {map_val:.1f} (114-129), Protein: {protein}+ (≥2)"
        return "Moderate", f"MAP: {map_val:.1f} (114-129), Protein: {protein} (below threshold)"
    elif map_val >= 130:
        if protein >= 2:
            return "Severe", f"MAP: {map_val:.1f} (≥130), Protein: {protein}+ (≥2)"
        return "Severe", f"MAP: {map_val:.1f} (≥130), Protein: {protein} (below threshold)"
    elif 101 <= map_val <= 106:
        return "Normal", f"MAP: {map_val:.1f} (101-106) - Borderline"
    return "Normal", f"MAP: {map_val:.1f} - Below thresholds"

def predict_severity():
    """Predict severity from input vitals"""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        encoder = joblib.load(ENCODER_PATH)
        
        with open(INPUT_FILE, "r") as f:
            values = list(map(float, f.read().split()))
            systolic, diastolic, protein = values[:3]
        
        map_val = (systolic + 2 * diastolic) / 3
        manual_severity, explanation = classify_manual(systolic, diastolic, protein, map_val)
        
        features = pd.DataFrame([[systolic, diastolic, protein, map_val]],
                              columns=['Systolic_BP', 'Diastolic_BP', 'Protein_Urine', 'MAP'])
        
        features_scaled = scaler.transform(features)
        pred = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0]
        ml_severity = encoder.inverse_transform([pred])[0]
        
        # Prepare output data for Kafka
        output_data = {
            'patientId': PATIENT_ID,
            'systolic': systolic,
            'diastolic': diastolic,
            'map': map_val,
            'proteinuria': int(protein),
            'temperature': values[3] if len(values) > 3 else 36.5,
            'heartRate': values[4] if len(values) > 4 else 75,
            'spo2': values[5] if len(values) > 5 else 98,
            'severity': manual_severity,
            'rationale': explanation,
            'mlSeverity': ml_severity,
            'mlProbability': {encoder.classes_[i]: float(proba[i]) for i in range(len(encoder.classes_))},
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Send to Kafka
        vitals_producer.send_vitals(output_data)
        
        print("\n" + "="*60)
        print("🔍 PREECLAMPSIA SEVERITY ASSESSMENT")
        print("="*60)
        print(f"📊 VITALS:")
        print(f"   • Blood Pressure: {systolic}/{diastolic} mmHg")
        print(f"   • Mean Arterial Pressure (MAP): {map_val:.1f} mmHg")
        print(f"   • Proteinuria: {protein}+ (0-4 scale)")
        if len(values) > 3:
            print(f"   • Body Temperature: {values[3]}°C")
        if len(values) > 4:
            print(f"   • Heart Rate: {values[4]} bpm")
        if len(values) > 5:
            print(f"   • Oxygen Saturation (SpO2): {values[5]}%")
        print()
        print(f"🏥 CLINICAL CLASSIFICATION:")
        print(f"   • Severity: {manual_severity}")
        print(f"   • Rationale: {explanation}")
        print()
        print(f"🤖 ML MODEL PREDICTION:")
        print(f"   • Predicted Severity: {ml_severity}")
        print(f"   • Probability Distribution:")
        for i, class_name in enumerate(encoder.classes_):
            print(f"     - {class_name}: {proba[i]:.1%}")
        print()
        
        if manual_severity == ml_severity:
            print("✅ Clinical and ML classifications AGREE")
        else:
            print("⚠️  Clinical and ML classifications DIFFER")
            print("   → Clinical guidelines take precedence for diagnosis")
        
        print("="*60)
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        if not all(os.path.exists(f) for f in [MODEL_PATH, SCALER_PATH, ENCODER_PATH]):
            print("Training new model...")
            train_and_save_model()
            predict_severity()

def generate_vitals():
    """Generate simulated patient data with realistic ranges"""
    map_category = random.choice(['normal', 'mild', 'moderate', 'severe'])
    
    if map_category == 'normal':
        map_val = random.uniform(70, 100)
        protein = random.choice([0, 0, 0, 1])
    elif map_category == 'mild':
        map_val = random.uniform(107, 113)
        protein = random.choice([0, 1, 1, 2])
    elif map_category == 'moderate':
        map_val = random.uniform(114, 129)
        protein = random.choice([1, 2, 2, 3])
    else:
        map_val = random.uniform(130, 160)
        protein = random.choice([2, 3, 3, 4])
    
    diastolic = random.randint(60, 110)
    systolic = int(3 * map_val - 2 * diastolic)
    systolic = max(100, min(200, systolic))
    
    temperature = round(random.uniform(36.0, 37.5), 1)
    heart_rate = random.randint(60, 100)
    spo2 = random.randint(95, 100)
    
    return {
        'systolic': systolic,
        'diastolic': diastolic,
        'protein': int(protein),
        'temperature': temperature,
        'heart_rate': heart_rate,
        'spo2': spo2
    }

if __name__ == "__main__":
    print("=== PREECLAMPSIA SEVERITY MONITORING ===")
    print("Using Clinical Guidelines")
    print("Real-time data streaming to Kafka")
    print()
    setup_environment()
    
    if not all(os.path.exists(f) for f in [MODEL_PATH, SCALER_PATH, ENCODER_PATH]):
        train_and_save_model()
    
    try:
        while True:
            vitals = generate_vitals()
            map_val = (vitals['systolic'] + 2 * vitals['diastolic']) / 3
            
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Patient {PATIENT_ID}:")
            print(f"  • BP: {vitals['systolic']}/{vitals['diastolic']} mmHg")
            print(f"  • MAP: {map_val:.1f} mmHg (calculated)")
            print(f"  • Proteinuria: {vitals['protein']}+ (0-4 scale)")
            print(f"  • Body Temperature: {vitals['temperature']}°C")
            print(f"  • Heart Rate: {vitals['heart_rate']} bpm")
            print(f"  • Oxygen Saturation (SpO2): {vitals['spo2']}%")
            
            with open(INPUT_FILE, 'w') as f:
                f.write(f"{vitals['systolic']} {vitals['diastolic']} {vitals['protein']} "
                        f"{vitals['temperature']} {vitals['heart_rate']} {vitals['spo2']}")
            
            predict_severity()
            time.sleep(INTERVAL_SECONDS)
            
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped.")



# C:\PregnancyMonitor\FinalYearProject-dbf78de82c34b695b6a120e5b07b6e4ebc3cd953>docker exec -it pm-kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic vitals-updates --from-beginning