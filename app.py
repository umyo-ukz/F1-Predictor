import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model/f1_model.pkl")

# Sample driver and constructor mappings
drivers = {
    "Lando Norris": 1,
    "Oscar Piastri": 2,
    "Charles Leclerc": 3,
    "Lewis Hamilton": 4,
    "George Russell": 5,
    "Kimi Antonelli": 6,
    "Max Verstappen": 7,
    "Yuki Tsunoda": 8,
    "Fernando Alonso": 9,
    "Lance Stroll": 10,
    "Franco Colapinto": 11,
    "Pierre Gasly": 12,
    "Oliver Bearman": 13,
    "Esteban Ocon": 14,
    "Isack Hadjar": 15,
    "Liam Lawson": 16,
    "Nico Hulkenberg": 17,
    "Gabriel Bortoleto": 18,
    "Alexander Albon": 19,
    "Carlos Sainz": 20
}

constructors = {
    "McLaren": 1,
    "Ferrari": 2,
    "Mercedes": 3,
    "Red Bull Racing": 4,
    "Aston Martin": 5,
    "Alpine": 6,
    "Haas": 7,
    "Racing Bulls": 8,
    "Sauber": 9,
    "Williams": 10
}

circuits = {
    "Australian Grand Prix": 1,
    "Chinese Grand Prix": 2,
    "Japanese Grand Prix": 3,
    "Bahrain Grand Prix": 4,
    "Saudi Arabian Grand Prix": 5,
    "Miami Grand Prix": 6,
    "Emilia Romagna Grand Prix": 7,
    "Monaco Grand Prix": 8,
    "Spanish Grand Prix": 9,
    "Canadian Grand Prix": 10,
    "Austrian Grand Prix": 11,
    "British Grand Prix": 12,
    "Belgian Grand Prix": 13,
    "Hungarian Grand Prix": 14,
    "Dutch Grand Prix": 15,
    "Italian Grand Prix": 16,
    "Azerbaijan Grand Prix": 17,
    "Singapore Grand Prix": 18,
    "United States Grand Prix": 19,
    "Mexico City Grand Prix": 20,
    "São Paulo Grand Prix": 21,
    "Las Vegas Grand Prix": 22,
    "Qatar Grand Prix": 23,
    "Abu Dhabi Grand Prix": 24
}

st.title("🏎️ F1 Top 3 Predictor")

driver = st.selectbox("Select Driver", list(drivers.keys()))
team = st.selectbox("Select Constructor", list(constructors.keys()))
circuit = st.selectbox("Select Circuit", list(circuits.keys()))
grid = st.slider("Starting Grid Position", 1, 20)

if st.button("Predict"):
    # Build input row to match training features exactly
    X_input = pd.DataFrame([[
        grid,
        drivers[driver],
        constructors[team],
        circuits[circuit]
    ]], columns=['grid', 'driverId', 'constructorId', 'circuitId'])
    
    prediction = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][1]

    if prediction:
        st.success(f"✅ {driver} is likely to finish in the Top 3! (Confidence: {prob:.2f})")
    else:
        st.info(f"ℹ️ {driver} is unlikely to finish Top 3. (Confidence: {prob:.2f})")
