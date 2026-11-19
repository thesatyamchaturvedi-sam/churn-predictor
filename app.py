import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉")


@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")


model = load_model()

st.title("📉 Customer Churn Predictor")
st.write("Enter customer details to estimate the probability they will leave.")

with st.form("customer"):
    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen", ["No", "Yes"])
        partner = st.selectbox("Has partner", ["No", "Yes"])
        dependents = st.selectbox("Has dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly = st.number_input("Monthly charges", 0.0, 200.0, 70.0, step=0.5)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment method",
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)", "Credit card (automatic)"],
        )
    with c2:
        phone = st.selectbox("Phone service", ["Yes", "No"])
        multi = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
        internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        no_net = internet == "No"
        net_opts = ["No internet service"] if no_net else ["No", "Yes"]
        security = st.selectbox("Online security", net_opts)
        backup = st.selectbox("Online backup", net_opts)
        protection = st.selectbox("Device protection", net_opts)
        support = st.selectbox("Tech support", net_opts)
        tv = st.selectbox("Streaming TV", net_opts)
        movies = st.selectbox("Streaming movies", net_opts)
    submitted = st.form_submit_button("Predict churn")

if submitted:
    row = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": "No phone service" if phone == "No" else multi,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        # the model was trained with TotalCharges as text, so pass a string
        "TotalCharges": str(round(tenure * monthly, 2)),
    }
    df = pd.DataFrame([row])
    classes = list(model.classes_)
    prob = float(model.predict_proba(df)[0][classes.index("Yes")])

    st.subheader(f"Churn probability: {prob:.1%}")
    st.progress(min(max(prob, 0.0), 1.0))
    if prob >= 0.5:
        st.error("High risk: this customer is likely to churn.")
    elif prob >= 0.3:
        st.warning("Medium risk: worth keeping an eye on.")
    else:
        st.success("Low risk: this customer is likely to stay.")
