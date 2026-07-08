# Import Required Libraries

import streamlit as st
import pandas as pd
import requests 



# 4. Configure Application Page

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# 5. Display Application Header

st.title("Customer Churn Prediction System")

st.write(
    "Predict customer churn risk using machine learning."
)


# 7. Customer Information Form

st.subheader("Enter Customer Details")

with st.form("customer_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            ["No", "Yes"]
        )

    with col2:
        partner = st.selectbox(
            "Partner",
            ["No", "Yes"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["No", "Yes"]
        )

    with col3:
        tenure = st.number_input(
            "Tenure (Months)",
            min_value=0,
            max_value=72,
            value=12
        )
        
    # 8. Service Information

    st.subheader("Service Information")

    col4, col5, col6 = st.columns(3)

    with col4:
        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )

    with col5:
        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
            "Online Security",
            ["No", "Yes", "No internet service"]
        )

    with col6:
        online_backup = st.selectbox(
            "Online Backup",
            ["No", "Yes", "No internet service"]
        )

        device_protection = st.selectbox(
            "Device Protection",
            ["No", "Yes", "No internet service"]
        )

    # 9. Additional Service Information

    st.subheader("Additional Services")

    col7, col8, col9 = st.columns(3)

    with col7:
        tech_support = st.selectbox(
            "Tech Support",
            ["No", "Yes", "No internet service"]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["No", "Yes", "No internet service"]
        )

    with col8:
        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["No", "Yes", "No internet service"]
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

    with col9:
        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    # 10. Billing Information

    st.subheader("Billing Information")

    col10, col11 = st.columns(2)

    with col10:
        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=50.0
        )

    with col11:
        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=500.0
        )

    submit_button = st.form_submit_button(
        "Predict Churn Risk"
    )

# 11. Prepare Customer Data for Prediction

if submit_button:
    # 11. Validate Customer Input

    validation_errors = []

    if phone_service == "No" and multiple_lines != "No phone service":
        validation_errors.append(
            "Multiple Lines must be 'No phone service' when Phone Service is No."
        )

    internet_features = {
        "Online Security": online_security,
        "Online Backup": online_backup,
        "Device Protection": device_protection,
        "Tech Support": tech_support,
        "Streaming TV": streaming_tv,
        "Streaming Movies": streaming_movies
    }

    if internet_service == "No":
        for feature_name, feature_value in internet_features.items():
            if feature_value != "No internet service":
                validation_errors.append(
                    f"{feature_name} must be 'No internet service' when Internet Service is No."
                )

    if validation_errors:
        for error in validation_errors:
            st.error(error)

        st.stop()

    customer_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [1 if senior_citizen == "Yes" else 0],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    # st.write("Customer Input Data:")
    # st.dataframe(customer_data)


    # 12. Send Customer Data to Flask API

    API_URL = "http://127.0.0.1:5000/predict"

    try:
        response = requests.post(
            API_URL,
            json=customer_data.iloc[0].to_dict(),
            timeout=10
        )

        if response.status_code == 200:

            result = response.json()
            
            prediction = result["prediction"]
            churn_probability = result["churn_probability"]
            churn_factors = result["factors_toward_churn"]
            retention_factors = result["factors_toward_retention"]

            st.subheader("Prediction Result")

            st.metric(
                "Churn Probability",
                f"{churn_probability * 100:.2f}%"
            )

            if prediction == "Churn":
                st.warning(
                    " Prediction: Customer is at risk of churn."
                )
            else:
                st.success(
                    " Prediction: Customer is not currently classified as churn risk."
                )
            st.subheader("Why This Prediction?")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Factors pushing toward churn")

                for factor in churn_factors:
                    st.write(f"🔴 {factor}")


            with col2:
                st.markdown("#### Factors pushing toward retention")

                for factor in retention_factors:
                    st.write(f"🟢 {factor}")

        else:
            st.error(
                f"Prediction failed: "
                f"{response.json().get('message', 'Unknown API error')}"
            )


    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to the prediction API. "
            " Flask backend is  not running."
        )

    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")