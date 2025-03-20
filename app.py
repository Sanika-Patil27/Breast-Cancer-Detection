import streamlit as st  
import joblib  
import numpy as np  

# Load the trained model  
model = joblib.load("models/breast_cancer_model.pkl")

# Title with Image  
st.image("mammograms.jpg", use_container_width=True) 
st.title("🔬 Breast Cancer Detection App")  
st.write("### Enter the required input features to predict cancer.")

# Sidebar  
st.sidebar.header("🩺 About the App")  
st.sidebar.write("This AI-based tool predicts whether a tumor is **Benign (Non-Cancerous)** or **Malignant (Cancerous)** based on medical features.")  
st.sidebar.write("### 🔹 Instructions:")  
st.sidebar.write("1️⃣ Enter feature values below.")  
st.sidebar.write("2️⃣ Click the **Predict** button.")  
st.sidebar.write("3️⃣ Get instant diagnosis results.")  

# Define selected feature names  
feature_names = [
    "mean concave points", "radius error", "area error", "compactness error", "worst radius",
    "worst texture", "worst perimeter", "worst area", "worst concavity", "worst concave points"
]

# Input fields  
st.write("## 📊 Enter Feature Values:")
features = []  # Store selected features
for feature in feature_names:
    val = st.number_input(f"{feature}", value=0.0)  # Restrict to 0.0 and above
    features.append(val)

input_data = np.array([features])  # Convert to NumPy array

# Predict button  
if st.button("🔍 Predict"):  
    # Make prediction  
    prediction = model.predict(input_data)  
    prediction_proba = model.predict_proba(input_data)[0]  # Get probability scores
    
    benign_prob = prediction_proba[1] * 100  # Convert to percentage
    malignant_prob = prediction_proba[0] * 100  # Convert to percentage
    
    # Display result with probabilities  
    if prediction[0] == 1:
        st.success(f"🟢 The tumor is **Benign (Non-Cancerous)** with **{benign_prob:.2f}%** confidence.")
    else:
        st.error(f"🔴 The tumor is **Malignant (Cancerous)** with **{malignant_prob:.2f}%** confidence.")