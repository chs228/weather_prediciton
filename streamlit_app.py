import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Function to fetch weather data from OpenWeatherMap API
def fetch_weather_data(city, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={vellore}&appid={8e053535e539905553c1c736985c5ccf}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {
            "Temperature": data["main"]["temp"],
            "Humidity": data["main"]["humidity"],
            "Pressure": data["main"]["pressure"],
            "Description": data["weather"][0]["description"],
        }
    else:
        return None

# Simulated dataset for training the ML model (replace with real historical data if available)
def generate_sample_data():
    np.random.seed(42)
    temperatures = np.random.uniform(15, 40, 1000)
    humidity = np.random.uniform(20, 80, 1000)
    pressure = np.random.uniform(1000, 1025, 1000)
    target_temp = temperatures + np.random.normal(0, 1, 1000)  # Adding some noise

    data = pd.DataFrame({
        "Temperature": temperatures,
        "Humidity": humidity,
        "Pressure": pressure,
        "TargetTemp": target_temp,
    })
    return data

# Train ML model
def train_model(data):
    X = data[["Humidity", "Pressure"]]
    y = data["TargetTemp"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model

# Streamlit app
st.title("Weather Prediction App")

# User input for city
city = st.text_input("Enter city name:", "Mumbai")
api_key = "YOUR_API_KEY"

# Fetch and display real-time weather data
if st.button("Get Weather Data"):
    weather_data = fetch_weather_data(city, api_key)
    if weather_data:
        st.write(f"### Current Weather in {city}")
        st.write(f"- Temperature: {weather_data['Temperature']} °C")
        st.write(f"- Humidity: {weather_data['Humidity']} %")
        st.write(f"- Pressure: {weather_data['Pressure']} hPa")
        st.write(f"- Description: {weather_data['Description']}")
    else:
        st.error("Failed to fetch weather data. Please check the city name or your API key.")

# ML-based temperature prediction
st.write("---")
st.write("### Predict Future Temperature")

# User input for prediction features
humidity = st.slider("Humidity (%)", 0, 100, 50)
pressure = st.slider("Pressure (hPa)", 950, 1050, 1013)

# Train the model and make a prediction
data = generate_sample_data()
model = train_model(data)

if st.button("Predict Temperature"):
    prediction = model.predict([[humidity, pressure]])[0]
    st.write(f"### Predicted Temperature: {prediction:.2f} °C")
