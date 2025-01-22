import requests
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import firebase_admin
from firebase_admin import credentials, db
import io
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# --- Firebase Initialization ---
# Initialize Firebase Admin SDK only if it's not already initialized
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase-adminsdk.json")  # Replace with your Firebase credentials file
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://weather-prediction-5ef4b-default-rtdb.firebaseio.com'  # Replace with your Firebase Realtime Database URL
    })

# --- API Configuration ---
API_KEY = "SK3M2MX6SE39DM7JJA2P2HZAU"  # Replace with your Visual Crossing API Key
LOCATION = "Vellore,India"

# --- Fetch Real-Time Weather Data ---
def fetch_weather():
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{LOCATION}?unitGroup=metric&contentType=csv&include=days&key={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        new_data = pd.read_csv(io.StringIO(response.text))
        return new_data
    else:
        print("Failed to fetch data:", response.text)
        return None

# --- Store Data to Firebase Realtime Database ---
def store_in_realtime_db(new_data):
    if new_data is not None and not new_data.empty:
        for _, row in new_data.iterrows():
            data = row.to_dict()  # Convert each row to a dictionary
            db.reference('/weather_data').push(data)  # Push data to the Firebase node

# --- Train Prediction Model ---
def train_model():
    # Fetch all data from Firebase Realtime Database
    ref = db.reference('/weather_data')
    data = ref.get()

    if data is None:
        print("No data available in Firebase.")
        return

    # Convert Firebase data into a DataFrame
    df = pd.DataFrame(data)

    features = ["humidity", "windgust", "windspeed"]
    target = "temp"

    df = df.dropna(subset=features + [target])  # Remove missing values
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Model Trained - MAE:", mean_absolute_error(y_test, y_pred))

    joblib.dump(model, "weather_model.pkl")  # Save the model

# --- Streamlit App ---
def weather_dashboard():
    st.title("🌦️ Vellore Weather Prediction App")
    st.write("This app predicts temperature based on historical weather data.")

    if st.button("Predict Temperature"):
        # Load trained model
        model = joblib.load("weather_model.pkl")
        
        # Use historical data from Firebase to train and predict
        ref = db.reference('/weather_data')
        data = ref.get()

        if data is None:
            st.write("No historical data available for prediction.")
            return

        df = pd.DataFrame(data)
        if df.empty:
            st.write("No data to make predictions.")
            return
        
        # Get the latest available data for prediction
        latest_data = df.iloc[-1]  # You can change this to select other data points
        humidity = latest_data['humidity']
        windgust = latest_data['windgust']
        windspeed = latest_data['windspeed']
        
        # Prepare the input data for prediction
        input_data = np.array([[humidity, windgust, windspeed]])
        
        # Make prediction
        prediction = model.predict(input_data)
        st.write(f"Predicted Temperature: {prediction[0]:.2f}°C")

# --- Main Execution ---
if __name__ == "__main__":
    new_data = fetch_weather()
    if new_data is not None:
        store_in_realtime_db(new_data)  # Store the new data in Firebase
        train_model()  # Train the model using the Firebase data

    weather_dashboard()  # Display Streamlit dashboard
