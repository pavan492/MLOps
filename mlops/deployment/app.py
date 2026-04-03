import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="Pavan-K492/tourism-package-prediction-model", filename="best_tourism-package-prediction-model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Tourism package Prediction
st.title("Tourism package Prediction App")
st.write("The Tourism package Prediction App is to predict whether a customer will purchase a package based on info developed for Visit with Us company.")
st.write("Kindly enter the details to check whether they are likely to buy a tourism package.")

# Collect user input
CityTier = st.selectbox("City Tier", ["1", "2" ,"3"])
NumberOfPersonVisiting = st.number_input("Number of People Visiting", min_value=1, value=1)
Passport = st.selectbox("Passport?", ["Yes", "No"])
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", min_value=1 , max_value=10, value=5)
OwnCar = st.selectbox("Own Car?", ["Yes", "No"])
Age = st.number_input("Age (customer's age in years)", min_value=18, max_value=100, value=30)
DurationOfPitch = st.number_input("Duration of Pitch (duration of the sales pitch in minutes)", min_value=1,value=1)
NumberOfFollowups =st.number_input("Number of Follow-ups (total number of follow-ups by the salesperson after the sales pitch)", min_value=1, value=1)
PreferredPropertyStar = st.number_input("Preferred Property Star (preferred hotel rating by the customer)", min_value=1, max_value=5, value=1)
NumberOfTrips = st.number_input("Number of Trips (average number of trips the customer takes annually)",min_value=1, value=1)
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting",min_value=1,max_value=5, value=1)
MonthlyIncome = st.number_input("Monthly Income (gross monthly income of the customer)", min_value=0, value=5000)
TypeofContact = st.selectbox("Type of Contact (the method by which the customer was contacted)", ["Company Invited", "Self Inquiry"])
Occupation = st.selectbox("Occupation (customer's occupation)", ["Salaried", "Free Lancer","Small Business","Large Business"])
Gender = st.selectbox("Gender (customer's gender)", ["Male", "Female"])
ProductPitched = st.selectbox("Product Pitched (the type of)", ["Basic", "Deluxe","Standard","Super Deluxe","King"])
MaritalStatus = st.selectbox("Marital Status (customer's marital status)", ["Single", "Married", "Divorced"])
Designation = st.selectbox("Designation (customer's designation in their current organization)", ["Executive", "Manager", "Senior Manager", "AVP","VP"])


# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'CityTier': CityTier,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'Designation': Designation,
    'MaritalStatus': MaritalStatus,
    'ProductPitched': ProductPitched,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'Gender': Gender,
    'NumberOfTrips': NumberOfTrips,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'Passport': 1 if Passport == "Yes" else 0,
    'Age': Age,
    'DurationOfPitch': DurationOfPitch,
    'NumberOfFollowups': NumberOfFollowups,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome': MonthlyIncome,
    'TypeofContact': 1 if TypeofContact == "Company Invited" else 0,
    'Occupation': Occupation
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "buy" if prediction == 1 else "not buy"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
