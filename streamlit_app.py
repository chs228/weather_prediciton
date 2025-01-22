import requests
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import io

# --- API Configuration ---
API_KEY = "SK3M2MX6SE39DM7JJA2P2HZAU"  # Replace with your Visual Crossing API Key
LOCATION = "Vellore,India"
CSV_FILE = "weather-prediction.csv"

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
    
    # Check if file exists and is not empty
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

    # Save the updated data
    updated_data.to_csv(CSV_FILE, index=False)
    print("Data updated successfully.")

# --- Train Prediction Model ---
def train_model():
    if not os.path.exists(CSV_FILE):
        print("CSV file not found.")
        return
    
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
    st.write("This app predicts temperature based on historical weather data.")

    if st.button("Predict Temperature"):
        # Load trained model
        model = joblib.load("weather_model.pkl")
        
        # Load historical data from CSV
        if not os.path.exists(CSV_FILE):
            st.write("No historical data available for prediction.")
            return

        df = pd.read_csv(CSV_FILE)
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
        append_to_csv(new_data)  # Store the new data in CSV
        train_model()  # Train the model using the CSV data

    weather_dashboard()  # Display Streamlit dashboard
