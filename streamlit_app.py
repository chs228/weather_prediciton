import requests
import pandas as pd
import numpy as np
import joblib
import time
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# --- API Configuration ---
API_KEY = "SK3M2MX6SE39DM7JJA2P2HZAU"  # Replace with your Visual Crossing API Key
LOCATION = "Vellore,India"
CSV_FILE = "weather_data.csv"

# --- Fetch Real-Time Weather Data ---
def fetch_weather():
    API_KEY = "SK3M2MX6SE39DM7JJA2P2HZAU"  # Replace with your Visual Crossing API Key
    LOCATION = "Vellore,India"
    CSV_FILE = "weather_data.csv"

    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{LOCATION}?unitGroup=metric&contentType=csv&include=days&key={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        new_data = pd.read_csv(io.StringIO(response.text))
        return new_data
    else:
        print("Failed to fetch data:", response.text)
        return None

# --- Append Data to CSV ---
def append_to_csv(new_data):
    try:
        existing_data = pd.read_csv(CSV_FILE)
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
    except FileNotFoundError:
        updated_data = new_data

    updated_data.to_csv(CSV_FILE, index=False)
    print("Data updated successfully.")

# --- Train Prediction Model ---
def train_model():
    df = pd.read_csv(CSV_FILE)
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
    st.write("Enter weather conditions to predict the temperature:")

    humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=50)
    windgust = st.number_input("Wind Gust (km/h)", min_value=0.0, max_value=100.0, value=10.0)
    windspeed = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=100.0, value=5.0)

    if st.button("Predict Temperature"):
        model = joblib.load("weather_model.pkl")
        input_data = np.array([[humidity, windgust, windspeed]])
        prediction = model.predict(input_data)
        st.write(f"Predicted Temperature: {prediction[0]:.2f}°C")

# --- Run Data Fetching, Training, and Streamlit ---
if __name__ == "__main__":
    new_data = fetch_weather()
    if new_data is not None:
        append_to_csv(new_data)
        train_model()
    
    weather_dashboard()
