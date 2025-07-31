# Import pandas library and filter only important data
import pandas as pd

def load_data():
    df = pd.read_csv("C:/Users/Administrator/Documents/VSC Projects/F1 Predictor/data/results.csv")
    df = df[['raceId', 'driverId', 'constructorId', 'grid', 'positionOrder', 'points']]
    df = df.dropna()
    return df

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def train_model():
    df = load_data()
    df['top3'] = df['positionOrder'] <= 3
    X = df[['grid', 'driverId', 'constructorId']] # Features, the data we use to predict
    y = df['top3'] # Target variable, what we want to predict


# Split the data into training and testing sets, 20% of data will be used for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Accuracy: {acc:.2f}")

    joblib.dump(model, 'C:/Users/Administrator/Documents/VSC Projects/F1 Predictor/model/f1_model.pkl')

if __name__ == "__main__":
    train_model()
    