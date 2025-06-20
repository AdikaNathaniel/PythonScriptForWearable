import time
import random
from datetime import datetime
import pandas as pd
import joblib
import os
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

def setup_environment():
    """Verify all files exist."""
    os.chdir(WORKING_DIR)
    print(f"✓ Working in: {WORKING_DIR}")
    
    if not os.path.exists("realistic_noisy_preeclampsia_dataset_noisy.csv"):
        raise FileNotFoundError("Missing CSV dataset file")

def calculate_severity(df):
    """Calculate severity levels based on clinical thresholds"""
    conditions = [
        (df['Systolic_BP'] < 140) & (df['Diastolic_BP'] < 90) & (df['Protein_Urine'] < 2),
        (df['Systolic_BP'] < 160) & (df['Diastolic_BP'] < 110) & (df['Protein_Urine'] < 3),
        (df['Systolic_BP'] >= 160) | (df['Diastolic_BP'] >= 110) | (df['Protein_Urine'] >= 3)
    ]
    choices = ['Mild', 'Moderate', 'Severe']
    return np.select(conditions, choices, default='Moderate')

def load_and_preprocess_data():
    """Load data and create severity labels"""
    df = pd.read_csv("realistic_noisy_preeclampsia_dataset_noisy.csv")
    
    # Create severity labels if not present
    if 'Severity' not in df.columns:
        df['Severity'] = calculate_severity(df)
    
    # Use existing MAP or calculate
    df['MAP'] = df.get('MAP', (df['Systolic_BP'] + 2 * df['Diastolic_BP']) / 3)
    
    # Encode severity
    encoder = LabelEncoder()
    df['Severity_Encoded'] = encoder.fit_transform(df['Severity'])
    
    # Select features
    features = ['Systolic_BP', 'Diastolic_BP', 'Protein_Urine', 'MAP']
    X = df[features]
    y = df['Severity_Encoded']
    
    return X, y, encoder, features

def train_and_save_model():
    """Train and save severity prediction model"""
    X, y, encoder, features = load_and_preprocess_data()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = LogisticRegression(multi_class='multinomial', max_iter=1000)
    model.fit(X_scaled, y)
    
    # Save artifacts
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    print("✅ Severity model trained and saved")
    return features

def predict_severity():
    """Predict severity from input vitals"""
    try:
        # Load models
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        encoder = joblib.load(ENCODER_PATH)
        
        # Read input
        with open(INPUT_FILE, "r") as f:
            systolic, diastolic, protein = map(float, f.read().split())
        
        # Calculate MAP
        map_val = (systolic + 2 * diastolic) / 3
        
        # Prepare features
        features = pd.DataFrame([[systolic, diastolic, protein, map_val]],
                              columns=['Systolic_BP', 'Diastolic_BP', 'Protein_Urine', 'MAP'])
        
        # Predict
        features_scaled = scaler.transform(features)
        pred = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0]
        
        # Format output
        print("\n" + "="*50)
        print("🔍 Severity Prediction Results:")
        print(f"BP: {systolic}/{diastolic} mmHg")
        print(f"Proteinuria: {protein} (0-4 scale)")
        print(f"Predicted Severity: {encoder.inverse_transform([pred])[0]}")
        print("Probability Distribution:")
        for i, class_name in enumerate(encoder.classes_):
            print(f"  {class_name}: {proba[i]:.1%}")
        print("="*50)
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        if not all(os.path.exists(f) for f in [MODEL_PATH, SCALER_PATH, ENCODER_PATH]):
            print("Training new model...")
            train_and_save_model()
            predict_severity()

def generate_vitals():
    """Generate simulated patient data"""
    return {
        'systolic': random.randint(100, 180),
        'diastolic': random.randint(60, 110),
        'protein': random.randint(0, 4)
    }

if __name__ == "__main__":
    print("=== Preeclampsia Severity Monitoring ===")
    setup_environment()
    
    # Train model if not exists
    if not all(os.path.exists(f) for f in [MODEL_PATH, SCALER_PATH, ENCODER_PATH]):
        train_and_save_model()
    
    try:
        while True:
            # Generate and display data
            vitals = generate_vitals()
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Patient {PATIENT_ID}:")
            print(f"  • BP: {vitals['systolic']}/{vitals['diastolic']} mmHg")
            print(f"  • Proteinuria: {vitals['protein']} (0-4 scale)")
            
            # Write input
            with open(INPUT_FILE, 'w') as f:
                f.write(f"{vitals['systolic']} {vitals['diastolic']} {vitals['protein']}")
            
            # Make prediction
            predict_severity()
            time.sleep(INTERVAL_SECONDS)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")