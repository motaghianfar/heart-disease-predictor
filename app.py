import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# Load models and scaler
@st.cache_resource
def load_models():
    models = {}
    try:
        models['logistic'] = pickle.load(open('heart_disease_model_logistic_regression.pkl', 'rb'))
        models['random_forest'] = pickle.load(open('heart_disease_model_random_forest.pkl', 'rb'))
        models['xgboost'] = pickle.load(open('heart_disease_model_xgboost.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        return models, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

def main():
    st.title("❤️ Heart Disease Prediction App")
    st.write("This app predicts the likelihood of heart disease based on medical parameters.")
    
    # Load models
    models, scaler = load_models()
    
    if models is None:
        st.error("Failed to load models. Please check if all model files are available.")
        return
    
    # Sidebar for input parameters
    st.sidebar.header("Patient Parameters")
    
    # Create input fields
    age = st.sidebar.slider("Age", 20, 100, 50)
    sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
    cp = st.sidebar.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
    chol = st.sidebar.slider("Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
    restecg = st.sidebar.selectbox("Resting ECG Results", [0, 1, 2])
    thalach = st.sidebar.slider("Maximum Heart Rate Achieved", 60, 220, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.sidebar.slider("ST Depression induced by exercise", 0.0, 6.0, 1.0)
    slope = st.sidebar.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])
    ca = st.sidebar.slider("Number of Major Vessels colored by fluoroscopy", 0, 4, 1)
    thal = st.sidebar.selectbox("Thalassemia", [1, 2, 3])
    
    # Convert categorical inputs to numerical
    sex_num = 1 if sex == "Male" else 0
    fbs_num = 1 if fbs == "Yes" else 0
    exang_num = 1 if exang == "Yes" else 0
    
    # Create feature array
    features = np.array([[age, sex_num, cp, trestbps, chol, fbs_num, restecg, thalach, exang_num, oldpeak, slope, ca, thal]])
    
    # Create engineered features (same as during training)
    age_category = 0 if age <= 45 else 1 if age <= 55 else 2 if age <= 65 else 3
    bp_category = 0 if trestbps <= 120 else 1 if trestbps <= 130 else 2 if trestbps <= 140 else 3
    chol_category = 0 if chol <= 200 else 1 if chol <= 240 else 2 if chol <= 300 else 3
    hr_category = 0 if thalach <= 100 else 1 if thalach <= 120 else 2 if thalach <= 140 else 3 if thalach <= 160 else 4 if thalach <= 180 else 5 if thalach <= 200 else 6
    bmi_risk = 1 if age > 50 else 0
    
    # Add engineered features to the feature array
    features_with_engineered = np.array([[age, sex_num, cp, trestbps, chol, fbs_num, restecg, thalach, exang_num, oldpeak, slope, ca, thal, 
                                        age_category, bp_category, chol_category, hr_category, bmi_risk]])
    
    # Scale features (only scale the original numerical features)
    numerical_features = features_with_engineered[:, :5]  # age, trestbps, chol, thalach, oldpeak
    scaled_numerical = scaler.transform(numerical_features)
    
    # Replace the numerical features with scaled ones
    features_scaled = features_with_engineered.copy()
    features_scaled[:, :5] = scaled_numerical
    
    # Prediction section
    st.header("Prediction Results")
    
    if st.button("Predict Heart Disease"):
        with st.spinner("Making predictions..."):
            # Make predictions with all models
            results = {}
            for model_name, model in models.items():
                prediction = model.predict(features_scaled)
                probability = model.predict_proba(features_scaled)
                results[model_name] = {
                    'prediction': prediction[0],
                    'probability': probability[0][1]  # Probability of heart disease
                }
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Logistic Regression")
            pred = results['logistic']['prediction']
            prob = results['logistic']['probability']
            st.metric(
                label="Prediction", 
                value="Heart Disease" if pred == 1 else "No Heart Disease",
                delta=f"{prob:.2%} probability"
            )
        
        with col2:
            st.subheader("Random Forest")
            pred = results['random_forest']['prediction']
            prob = results['random_forest']['probability']
            st.metric(
                label="Prediction", 
                value="Heart Disease" if pred == 1 else "No Heart Disease",
                delta=f"{prob:.2%} probability"
            )
        
        with col3:
            st.subheader("XGBoost")
            pred = results['xgboost']['prediction']
            prob = results['xgboost']['probability']
            st.metric(
                label="Prediction", 
                value="Heart Disease" if pred == 1 else "No Heart Disease",
                delta=f"{prob:.2%} probability"
            )
        
        # Show consensus
        st.subheader("Consensus Analysis")
        positive_count = sum(1 for result in results.values() if result['prediction'] == 1)
        total_models = len(results)
        
        if positive_count == total_models:
            st.error("⚠️ All models indicate HIGH risk of heart disease")
        elif positive_count >= total_models / 2:
            st.warning("⚠️ Majority of models indicate MODERATE risk of heart disease")
        else:
            st.success("✅ Majority of models indicate LOW risk of heart disease")

if __name__ == "__main__":
    main()
