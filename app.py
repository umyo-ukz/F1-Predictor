import streamlit as st # Framework for building web apps
import pandas as pd # Library for data manipulation
import joblib # Library for loading the trained model

# Load model
model = joblib.load("model/f1_model.pkl")
# Sample driver and constructor mappings
drivers = {
    "Max Verstappen": 1,
    "Lewis Hamilton": 2,
    "Charles Leclerc": 3
}

constructors = {
    "Red Bull": 1,
    "Mercedes": 2,
    "Ferrari": 3
}

st.title("🏁 F1 Top 3 Predictor")

driver = st.selectbox("Select Driver", list(drivers.keys()))
team = st.selectbox("Select Constructor", list(constructors.keys()))
grid = st.slider("Starting Grid Position", 1, 20)

if st.button("Predict"):
    X_input = pd.DataFrame([[grid, drivers[driver], constructors[team]]],
                           columns=['grid', 'driverId', 'constructorId'])
    
    prediction = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][1]

    if prediction:
        st.success(f"{driver} is likely to finish in the Top 3! 🔥 (Confidence: {prob:.2f})")
    else:
        st.info(f"{driver} is unlikely to finish Top 3. (Confidence: {prob:.2f})")
