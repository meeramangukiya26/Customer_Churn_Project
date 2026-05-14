import streamlit as st
# import pickle

# # Load model
# model = pickle.load(open("churn_model.pkl", "rb"))
import joblib

model = joblib.load("churn_model.pkl")

# Title
st.title("📊 Customer Churn Prediction")
st.image("customer.jpg")

st.markdown(
    """
    <style>
    .stApp {
        background-color: lightblue;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.title("About Project")

st.sidebar.write(
    "This is Customer Churn Prediction ML Project using Random Forest."
)
# Inputs
gender = st.number_input("Gender")
senior = st.number_input("SeniorCitizen")
partner = st.number_input("Partner")
dependents = st.number_input("Dependents")
tenure = st.number_input("Tenure")
phoneservice = st.number_input("PhoneService")
multiplelines = st.number_input("MultipleLines")
internetservice = st.number_input("InternetService")
onlinesecurity = st.number_input("OnlineSecurity")
onlinebackup = st.number_input("OnlineBackup")
deviceprotection = st.number_input("DeviceProtection")
techsupport = st.number_input("TechSupport")
streamingtv = st.number_input("StreamingTV")
streamingmovies = st.number_input("StreamingMovies")
contract = st.number_input("Contract")
paperlessbilling = st.number_input("PaperlessBilling")
paymentmethod = st.number_input("PaymentMethod")
monthlycharges = st.number_input("MonthlyCharges")
totalcharges = st.number_input("TotalCharges")

# Predict button
if st.button("Predict"):

    data = [[
        gender,
        senior,
        partner,
        dependents,
        tenure,
        phoneservice,
        multiplelines,
        internetservice,
        onlinesecurity,
        onlinebackup,
        deviceprotection,
        techsupport,
        streamingtv,
        streamingmovies,
        contract,
        paperlessbilling,
        paymentmethod,
        monthlycharges,
        totalcharges
    ]]

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("❌ Customer Will Churn")
    else:
        st.success("✅ Customer Will Stay")

