import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Set base path so it works locally and on deployment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(BASE_DIR, "data", "results.csv")
RACES_PATH = os.path.join(BASE_DIR, "data", "races.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "f1_model.pkl")

def load_data():
    """Load and preprocess the F1 dataset."""
    # Load results and races
    results = pd.read_csv(RESULTS_PATH)
    races = pd.read_csv(RACES_PATH)

    # Merge results with circuitId from races
    df = results.merge(
        races[['raceId', 'circuitId']],
        on="raceId",
        how="left"
    )

    # Keep relevant columns
    df = df[['raceId', 'driverId', 'constructorId', 'grid', 'positionOrder', 'points', 'circuitId']]
    df = df.dropna()

    # Binary target: True if driver finished in top 3, else False
    df['top3'] = df['positionOrder'] <= 3

    return df

def preprocess_data(df):
    """Encode categorical IDs into numeric labels."""
    le_driver = LabelEncoder()
    le_constructor = LabelEncoder()
    le_circuit = LabelEncoder()

    df['driverId'] = le_driver.fit_transform(df['driverId'])
    df['constructorId'] = le_constructor.fit_transform(df['constructorId'])
    df['circuitId'] = le_circuit.fit_transform(df['circuitId'])

    return df, le_driver, le_constructor, le_circuit

def train_model():
    """Train the Random Forest model and save it to disk."""
    df = load_data()
    df, le_driver, le_constructor, le_circuit = preprocess_data(df)

    # Features: grid position, driver, constructor, and circuit
    X = df[['grid', 'driverId', 'constructorId', 'circuitId']]
    y = df['top3']

    # Split data into training and test sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate accuracy
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Model Accuracy: {acc:.2f}")

    # Save model and encoders
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({
        "model": model,
        "le_driver": le_driver,
        "le_constructor": le_constructor,
        "le_circuit": le_circuit
    }, MODEL_PATH)
    print(f"Model and encoders saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
