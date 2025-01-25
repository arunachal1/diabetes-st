import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

# Load the model
model = joblib.load('./models/logreg_model.joblib')

# Define the API endpoint
@st.experimental_singleton
def predict(data):
    input_df = pd.DataFrame(data)
    prediction = model.predict(input_df)
    result = "Diabetes" if prediction[0] == 1 else "Not Diabetes"
    return result

# Define the Streamlit app
def app():
    st.title("Diabetes Prediction App")
    st.write("Enter the details below to make a prediction.")

    pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
    glucose = st.number_input("Glucose", min_value=0, step=1)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, step=1)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, step=1)
    insulin = st.number_input("Insulin", min_value=0, step=1)
    bmi = st.number_input("BMI", min_value=0.0, step=0.1)
    diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, step=0.1)
    age = st.number_input("Age", min_value=0, step=1)

    if st.button("Predict"):
        input_data = {
            "Pregnancies": [pregnancies],
            "Glucose": [glucose],
            "BloodPressure": [blood_pressure],
            "SkinThickness": [skin_thickness],
            "Insulin": [insulin],
            "BMI": [bmi],
            "DiabetesPedigreeFunction": [diabetes_pedigree_function],
            "Age": [age]
        }
        result = predict(input_data)
        st.success(f"Prediction: {result}")

if __name__ == "__main__":
    app()
