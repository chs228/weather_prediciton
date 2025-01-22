import requests
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import io
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# --- API Configuration ---
API_KEY = "SK3M2MX6SE39DM7JJA2P2HZAU"  # Replace with your Visual Crossing API Key
LOCATION = "Vellore,India"
CSV_FILE = "weather_data.csv"

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

# --- Append Data to CSV ---
def append_to_csv(new_data):
    if new_data is None or new_data.empty:
        print("No new data to append.")
        return
    
    if os.path.exists(CSV_FILE) and os.path.getsize(CSV_FILE) > 0:
        try:
            existing_data = pd.read_csv(CSV_FILE)
            updated_data = pd.concat([existing_data, new_data], ignore_index=True)
        except pd.errors.EmptyDataError:
            print("Warning: Existing CSV file is empty. Writing new data from scratch.")
            updated_data = new_data
    else:
        print("CSV file not found or empty. Creating new file.")
        updated_data = new_data

    updated_data.to_csv(CSV_FILE, index=False)
    print("Data updated successfully.")

# --- Train Prediction Model ---
def train_model():
    df = pd.read_csv(CSV_FILE)
    features = ["humidity", "windgust", "windspeed"]
    target = "temp"
    
    df = df.dropna(subset=features + [target])  
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("Model Trained - MAE:", mean_absolute_error(y_test, y_pred))
    
    joblib.dump(model, "weather_model.pkl")

# --- Streamlit App ---
def weather_dashboard():
    st.title("🌦️ Vellore Weather Prediction App")
    
    if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
        st.error("No data available. Please fetch data first.")
        return

    df = pd.read_csv(CSV_FILE)
    
    if df.empty or any(col not in df.columns for col in ["humidity", "windgust", "windspeed"]):
        st.error("Insufficient data for prediction.")
        return

    # Use the most recent row for prediction
    latest_data = df[["humidity", "windgust", "windspeed"]].iloc[-1:].values

    model = joblib.load("weather_model.pkl")
    prediction = model.predict(latest_data)

    st.subheader("Latest Weather Data")
    st.write(df.iloc[-1])  # Display latest row

    st.subheader("Predicted Temperature")
    st.write(f"🌡️ {prediction[0]:.2f}°C")

# --- Run Data Fetching, Training, and Streamlit ---
if __name__ == "__main__":
    new_data = fetch_weather()
    if new_data is not None:
        append_to_csv(new_data)
        train_model()
    
    weather_dashboard()
