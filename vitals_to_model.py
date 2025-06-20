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
    """Calculate severity levels based on clinical thresholds from University of Ghana guidelines"""
    # First check if MAP indicates any level of preeclampsia
    # If MAP is in normal range (70-100), it's not preeclampsia regardless of protein
    
    conditions = [
        # Normal/Low - MAP 70-100, Protein NIL
        (df['MAP'] >= 70) & (df['MAP'] <= 100) & (df['Protein_Urine'] == 0),
        
        # Mild Preeclampsia - MAP 107-113, Protein 1+
        (df['MAP'] >= 107) & (df['MAP'] <= 113) & (df['Protein_Urine'] >= 1),
        
        # Moderate Preeclampsia - MAP 114-129, Protein 2+ OR >
        (df['MAP'] >= 114) & (df['MAP'] <= 129) & (df['Protein_Urine'] >= 2),
        
        # High/Severe Preeclampsia - MAP 130 OR >, Protein 2+ OR >
        (df['MAP'] >= 130) & (df['Protein_Urine'] >= 2),
        
        # Cases where MAP is elevated but protein doesn't match - classify by MAP level
        # Mild range MAP but protein doesn't match
        (df['MAP'] >= 107) & (df['MAP'] <= 113) & (df['Protein_Urine'] < 1),
        
        # Moderate range MAP but protein doesn't match
        (df['MAP'] >= 114) & (df['MAP'] <= 129) & (df['Protein_Urine'] < 2),
    ]
    
    choices = [
        'Normal',      # Normal MAP and protein
        'Mild',        # Mild preeclampsia
        'Moderate',    # Moderate preeclampsia  
        'Severe',      # Severe preeclampsia
        'Mild',        # Mild range MAP (even with low protein)
        'Moderate'     # Moderate range MAP (even with low protein)
    ]
    
    # Default case: if MAP is normal (70-100) but protein is elevated, it's not preeclampsia
    result = np.select(conditions, choices, default='Normal')
    
    # Special handling: If MAP is in normal range (70-100), always classify as Normal
    # regardless of protein levels (as per your note)
    normal_map_mask = (df['MAP'] >= 70) & (df['MAP'] <= 100)
    result = np.where(normal_map_mask, 'Normal', result)
    
    return result

def load_and_preprocess_data():
    """Load data and create severity labels based on University of Ghana guidelines"""
    df = pd.read_csv("realistic_noisy_preeclampsia_dataset_noisy.csv")
    
    # Use existing MAP or calculate
    df['MAP'] = df.get('MAP', (df['Systolic_BP'] + 2 * df['Diastolic_BP']) / 3)
    
    # Create severity labels using the new classification
    df['Severity'] = calculate_severity(df)
    
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
    model = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42)
    model.fit(X_scaled, y)
    
    # Save artifacts
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    print("✅ Severity model trained and saved using University of Ghana guidelines")
    return features

def classify_manual(systolic, diastolic, protein, map_val):
    """Manual classification based on University of Ghana guidelines"""
    # Rule: If MAP is normal (70-100), it's not preeclampsia regardless of protein
    if 70 <= map_val <= 100:
        return "Normal", "MAP in normal range (70-100)"
    
    # Rule: MAP must be elevated for preeclampsia consideration
    elif 107 <= map_val <= 113:
        if protein >= 1:
            return "Mild", f"MAP: {map_val:.1f} (107-113), Protein: {protein}+ (≥1)"
        else:
            return "Mild", f"MAP: {map_val:.1f} (107-113), Protein: {protein} (below threshold but MAP elevated)"
    
    elif 114 <= map_val <= 129:
        if protein >= 2:
            return "Moderate", f"MAP: {map_val:.1f} (114-129), Protein: {protein}+ (≥2)"
        else:
            return "Moderate", f"MAP: {map_val:.1f} (114-129), Protein: {protein} (below threshold but MAP elevated)"
    
    elif map_val >= 130:
        if protein >= 2:
            return "Severe", f"MAP: {map_val:.1f} (≥130), Protein: {protein}+ (≥2)"
        else:
            return "Severe", f"MAP: {map_val:.1f} (≥130), Protein: {protein} (below threshold but MAP critically elevated)"
    
    elif 101 <= map_val <= 106:
        return "Normal", f"MAP: {map_val:.1f} (101-106) - Borderline but below preeclampsia threshold"
    
    else:
        return "Normal", f"MAP: {map_val:.1f} - Below preeclampsia thresholds"

def predict_severity():
    """Predict severity from input vitals using both ML model and clinical rules"""
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
        
        # Manual classification using clinical guidelines
        manual_severity, explanation = classify_manual(systolic, diastolic, protein, map_val)
        
        # ML model prediction
        features = pd.DataFrame([[systolic, diastolic, protein, map_val]],
                              columns=['Systolic_BP', 'Diastolic_BP', 'Protein_Urine', 'MAP'])
        
        features_scaled = scaler.transform(features)
        pred = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0]
        ml_severity = encoder.inverse_transform([pred])[0]
        
        # Format output
        print("\n" + "="*60)
        print("🔍 PREECLAMPSIA SEVERITY ASSESSMENT")
        print("="*60)
        print(f"📊 VITALS:")
        print(f"   • Blood Pressure: {systolic}/{diastolic} mmHg")
        print(f"   • Mean Arterial Pressure (MAP): {map_val:.1f} mmHg")
        print(f"   • Proteinuria: {protein}+ (0-4 scale)")
        print()
        print(f"🏥 CLINICAL CLASSIFICATION (UG Guidelines):")
        print(f"   • Severity: {manual_severity}")
        print(f"   • Rationale: {explanation}")
        print()
        print(f"🤖 ML MODEL PREDICTION:")
        print(f"   • Predicted Severity: {ml_severity}")
        print(f"   • Probability Distribution:")
        for i, class_name in enumerate(encoder.classes_):
            print(f"     - {class_name}: {proba[i]:.1%}")
        print()
        
        # Agreement check
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
    # Generate more realistic combinations
    map_category = random.choice(['normal', 'mild', 'moderate', 'severe'])
    
    if map_category == 'normal':
        map_val = random.uniform(70, 100)
        protein = random.choice([0, 0, 0, 1])  # Mostly 0, occasionally 1
    elif map_category == 'mild':
        map_val = random.uniform(107, 113)
        protein = random.choice([0, 1, 1, 2])  # Mix of protein levels
    elif map_category == 'moderate':
        map_val = random.uniform(114, 129)
        protein = random.choice([1, 2, 2, 3])  # Higher protein levels
    else:  # severe
        map_val = random.uniform(130, 160)
        protein = random.choice([2, 3, 3, 4])  # High protein levels
    
    # Back-calculate systolic and diastolic from MAP
    # MAP = (Systolic + 2*Diastolic) / 3
    # Use typical pulse pressure relationships
    diastolic = random.randint(60, 110)
    systolic = int(3 * map_val - 2 * diastolic)
    
    # Ensure realistic ranges
    systolic = max(100, min(200, systolic))
    
    return {
        'systolic': systolic,
        'diastolic': diastolic,
        'protein': int(protein)
    }

if __name__ == "__main__":
    print("=== PREECLAMPSIA SEVERITY MONITORING ===")
    print("Using University of Ghana Clinical Guidelines")
    print()
    setup_environment()
    
    # Train model if not exists
    if not all(os.path.exists(f) for f in [MODEL_PATH, SCALER_PATH, ENCODER_PATH]):
        train_and_save_model()
    
    try:
        while True:
            # Generate and display data
            vitals = generate_vitals()
            map_val = (vitals['systolic'] + 2 * vitals['diastolic']) / 3
            
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Patient {PATIENT_ID}:")
            print(f"  • BP: {vitals['systolic']}/{vitals['diastolic']} mmHg")
            print(f"  • MAP: {map_val:.1f} mmHg")
            print(f"  • Proteinuria: {vitals['protein']}+ (0-4 scale)")
            
            # Write input
            with open(INPUT_FILE, 'w') as f:
                f.write(f"{vitals['systolic']} {vitals['diastolic']} {vitals['protein']}")
            
            # Make prediction
            predict_severity()
            time.sleep(INTERVAL_SECONDS)
            
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped.")