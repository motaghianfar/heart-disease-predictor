import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# Custom unpickler to handle compatibility issues
class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Handle numpy core module changes
        if module == "numpy.core" or module.startswith("numpy.core."):
            module = module.replace("numpy.core.", "numpy.")
        if module == "numpy" and name == "ndarray":
            return np.ndarray
        if module == "numpy" and name == "dtype":
            return np.dtype
        if module == "numpy" and name == "array":
            return np.array
        if module == "numpy" and name == "float64":
            return np.float64
        if module == "numpy" and name == "int64":
            return np.int64
        return super().find_class(module, name)

def safe_pickle_load(file_path):
    """Safely load pickle files with compatibility handling"""
    try:
        with open(file_path, 'rb') as f:
            return SafeUnpickler(f).load()
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return None

# Load models and scaler
@st.cache_resource
def load_models():
    models = {}
    try:
        # Try loading with compatibility handling
        models['logistic'] = safe_pickle_load('heart_disease_model_logistic_regression.pkl')
        models['random_forest'] = safe_pickle_load('heart_disease_model_random_forest.pkl')
        models['xgboost'] = safe_pickle_load('heart_disease_model_xgboost.pkl')
        scaler = safe_pickle_load('scaler.pkl')
        
        # Check if all models loaded successfully
        if all(model is not None for model in models.values()) and scaler is not None:
            return models, scaler
        else:
            return None, None
            
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

def main():
    st.title("❤️ Heart Disease Prediction App")
    st.write("This app predicts the likelihood of heart disease based on medical parameters.")
    
    # Load models
    models, scaler = load_models()
    
    if models is None or scaler is None:
        st.error("Failed to load models. Please check if all model files are available.")
        st.info("""
        **Troubleshooting tips:**
        1. Make sure all .pkl files are in the same directory as app.py
        2. The models might have been created with incompatible library versions
        3. Try running the app in a clean environment with only the required packages
        """)
        
        # Show available files
        import os
        st.write("**Files in current directory:**")
        files = os.listdir('.')
        pkl_files = [f for f in files if f.endswith('.pkl')]
        for file in pkl_files:
            st.write(f"- {file}")
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
    
    # Prediction section
    st.header("Prediction Results")
    
    if st.button("Predict Heart Disease"):
        with st.spinner("Making predictions..."):
            try:
                # Scale features
                features_scaled = scaler.transform(features)
                
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
                    
            except Exception as e:
                st.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()
