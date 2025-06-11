import streamlit as st
import joblib
import pandas as pd

# Load model and feature names
model, feature_names, scaler = joblib.load(r"C:\Users\abhin\OneDrive\Desktop\Latedelivery\late_delivery_model (2).pkl")


st.title("🚚 LatePlate: Late Delivery Predictor")
st.write("Enter the order details below to check if the delivery is likely to be late.")

# Collect user input
distance = st.slider("Distance (km)", 1, 20, 5)
prep_time = st.slider("Preparation Time (min)", 5, 60, 10)
experience = st.slider("Courier Experience (yrs)", 0, 10, 2)

weather = st.selectbox("Weather", ['Sunny', 'Rainy', 'Snowy', 'Foggy', 'Windy'])
traffic = st.selectbox("Traffic Level", ['Low', 'Medium', 'High'])
time_of_day = st.selectbox("Time of Day", ['Morning', 'Evening', 'Night'])
vehicle = st.selectbox("Vehicle Type", ['Car', 'Scooter'])

# Raw input dictionary
input_dict = {
    'Distance_km': distance,
    'Preparation_Time_min': prep_time,
    'Courier_Experience_yrs': experience,
    'Weather': weather,
    'Traffic_Level': traffic,
    'Time_of_Day': time_of_day,
    'Vehicle_Type': vehicle
}

input_df = pd.DataFrame([input_dict])

# One-hot encode
input_encoded = pd.get_dummies(input_df)

# Fill missing columns
for col in feature_names:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# Reorder to match training
input_encoded = input_encoded[feature_names]

# Scale numerical columns
numerical = ['Distance_km', 'Courier_Experience_yrs', 'Preparation_Time_min']
input_encoded[numerical] = scaler.transform(input_encoded[numerical])

# Prediction
if st.button("🔍 Predict Late Delivery"):
    pred = model.predict(input_encoded)[0]
    prob = model.predict_proba(input_encoded)[0][1]

    if pred == 1:
        st.error(f"🚨 Delivery is **likely to be LATE**.\n\n📊 Probability: **{prob:.2f}**")
    else:
        st.success(f"✅ Delivery is **likely to be ON TIME**.\n\n📊 Probability of being late: **{prob:.2f}**")