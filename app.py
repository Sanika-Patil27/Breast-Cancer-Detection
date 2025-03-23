    
import joblib
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

# Load the trained model and scaler
model = joblib.load("breast_cancer_model.pkl")  # Ensure correct file path
scaler = joblib.load("breast_cancer_scaler.pkl")  # Ensure correct file path

# Retrieve feature names from the scaler (if available)
if hasattr(scaler, "feature_names_in_"):
    feature_names = list(scaler.feature_names_in_)
else:
    feature_names = [
        "mean radius", "worst concavity", "mean area", "mean concavity", "mean perimeter",
        "worst perimeter", "worst radius", "mean concave points", "worst concave points", "worst area"
    ]

# Load and display the image
image = Image.open("mammograms.jpg")
st.image(image, use_container_width=True)

# Sidebar with instructions
st.sidebar.title("🩺 About the App")
st.sidebar.write("This **AI-powered** tool predicts whether a tumor is **Benign** or **Malignant** based on medical features.")
st.sidebar.write("### 🔹 How to Use:")
st.sidebar.write("1️⃣ Enter the feature values in the input fields.")
st.sidebar.write("2️⃣ Click the **Predict** button.")
st.sidebar.write("3️⃣ View the **prediction result and probability.**")

# Main title
st.title("🔬 Breast Cancer Prediction App")
st.write("Enter the feature values below to predict whether a tumor is **Benign** or **Malignant**.")

# Collect user inputs
def get_user_input():
    user_data = {}
    st.write("### 📊 Enter Feature Values:")
    
    for feature in feature_names:
        user_data[feature] = st.number_input(f"{feature}") 
    
    return pd.DataFrame([user_data])

# Get user input
input_df = get_user_input()

# Display the entered data
st.write("### 📝 Your Entered Data:")
st.data_editor(input_df, use_container_width=True, num_rows="fixed")  # Editable table

# Predict button
if st.button("🔍 Predict", use_container_width=True):
    try:
        # Convert input data to numpy array and scale it
        input_array = input_df.astype(float).to_numpy().reshape(1, -1)
        scaled_input = scaler.transform(input_array)

        # Make prediction
        prediction = model.predict(scaled_input)
        prediction_proba = model.predict_proba(scaled_input)

        # Display result
        st.markdown("---")
        if prediction[0] == 0:
            st.error("🔴 **Malignant Tumor Detected! Please consult a doctor.**")
        else:
            st.success("🟢 **Benign Tumor Detected! No immediate concern.**")

        # Display prediction probability
        st.write("### 📊 Prediction Probability:")
        st.write(f"🔴 **Malignant:** {prediction_proba[0][0] * 100:.2f}%")
        st.write(f"🟢 **Benign:** {prediction_proba[0][1] * 100:.2f}%")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
