# Import required libraries
import streamlit as st
import pandas as pd
import joblib

# Load the saved model and scaler
model = joblib.load("models/linear_regression_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Application title
st.title("Medical Insurance Cost Prediction")

st.write("Enter customer details to predict medical insurance charges.")

# Take user inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)

sex = st.selectbox("Sex", ["female", "male"])

bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)

children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)

smoker = st.selectbox("Smoker", ["no", "yes"])

region = st.selectbox(
    "Region",
    ["northeast", "northwest", "southeast", "southwest"]
)

# Predict insurance charges when button is clicked
if st.button("Predict Insurance Cost"):

    # Convert categorical inputs into numerical values
    sex_value = 1 if sex == "male" else 0
    smoker_value = 1 if smoker == "yes" else 0

    # Prepare input data in the same order used during model training
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex_value],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker_value],
        'region_northeast': [1 if region == "northeast" else 0],
        'region_northwest': [1 if region == "northwest" else 0],
        'region_southeast': [1 if region == "southeast" else 0],
        'region_southwest': [1 if region == "southwest" else 0]
    })

    # Scale the input data
    scaled_input = scaler.transform(input_data)

    # Predict insurance charges
    prediction = model.predict(scaled_input)

    # Display predicted insurance cost
    st.success(f"Predicted Medical Insurance Cost: ₹{prediction[0]:,.2f}")