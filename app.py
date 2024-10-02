import numpy as np
import pandas as pd
import pickle
import streamlit as st

# Load the trained Random Forest model
with open("rf_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Function to predict medical costs
def predict_medical_cost(age, sex, bmi, children, smoker, region):
    # Prepare the input data
    input_data = np.array([[age, sex, bmi, children, smoker, region]])
    # Make the prediction
    prediction = model.predict(input_data)
    return prediction[0]  # Return the predicted charges

def main():
    st.title("Medical Insurance Cost Prediction")
    html_temp = """
    <div style="background-color:green;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Medical Insurance Prediction App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Create input fields for user input
    age = st.number_input("Age", min_value=0, value=25)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 1 else "Male")
    bmi = st.number_input("BMI", min_value=0.0, value=22.0)
    children = st.selectbox("Number of Children", options=[0, 1, 2, 3, 4, 5])
    smoker = st.selectbox("Smoker", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    region = st.selectbox("Region", options=[1, 2, 3, 4], format_func=lambda x: ["Southwest", "Southeast", "Northwest", "Northeast"][x-1])

    # Make prediction when the button is pressed
    if st.button("Predict"):
        result = predict_medical_cost(age, sex, bmi, children, smoker, region)
        st.success(f'The predicted medical cost is: ${result:.2f}')

    if st.button("About"):
        st.text("This app predicts medical insurance costs based on user inputs.")

if __name__ == '__main__':
    main()
