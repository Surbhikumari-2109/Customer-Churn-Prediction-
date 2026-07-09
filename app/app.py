# ============================================================
# CUSTOMER CHURN PREDICTION SYSTEM - STREAMLIT FRONTEND
# ============================================================

import streamlit as st
import pandas as pd
import requests


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #f7f8fc;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Hero Section */
.hero {
    background: #ffffff;
    padding: 22px 24px;
    border-radius: 14px;
    margin-bottom: 28px;
    text-align: center;

    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
}

.hero-title {
    color: #172554;
    font-size:34px;;
    font-weight: 700;
    margin: 0;
    text-align: center;
}

.hero-subtitle {
    color: #64748b;
    font-size: 14px;
    margin: 7px 0 0 0;
    line-height: 1.5;
    text-align: center;
}

    /* Section Headings */

    .section-title {
        color: #172554;
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .section-description {
        color: #64748b;
        font-size: 14px;
        margin-bottom: 16px;
    }


    /* Form */

    [data-testid="stForm"] {
        background-color: white;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 25px 28px 30px 28px;

        box-shadow:
            0 4px 20px
            rgba(15, 23, 42, 0.05);
    }


    /* Input Labels */

    [data-testid="stWidgetLabel"] p {
        font-weight: 600;
        color: #334155;
    }


    /* Submit Button */

    [data-testid="stFormSubmitButton"] button {
        width: 100%;
        height: 48px;
        background: #1e3a8a;
        color: white;
        border: none;
        border-radius: 10px;
        font-size: 16px;
        font-weight: 600;
    }

    [data-testid="stFormSubmitButton"] button:hover {
        background: #1d4ed8;
        color: white;
        border: none;
    }


    /* Probability Card */

    .probability-card {
        background-color: white;

        border: 1px solid #e5e7eb;
        border-radius: 16px;

        padding: 22px 25px;
        margin-bottom: 16px;

        box-shadow:
            0 4px 18px
            rgba(15, 23, 42, 0.05);
    }

    .probability-label {
        color: #64748b;
        font-size: 14px;
        font-weight: 600;

        text-transform: uppercase;
        letter-spacing: 0.6px;
    }

    .probability-value {
        color: #172554;
        font-size: 38px;
        font-weight: 750;
        margin-top: 3px;
    }


    /* Factor headings */

    .churn-heading {
        color: #b91c1c;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 12px;
    }

    .retention-heading {
        color: #047857;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 12px;
    }


    /* Footer */

    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 13px;
        padding-top: 35px;
    }
    /* Top Streamlit bar hide */
[data-testid="stHeader"] {
    display: none;
}

/* Page ko top se proper spacing */
.block-container {
    padding-top: 1.5rem;
}

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. HERO SECTION
# ============================================================

st.markdown(
    """
<div class="hero">
<div class="hero-title">Customer Churn Prediction System</div>
<div class="hero-subtitle">Estimate customer churn risk and understand the key factors influencing each prediction using machine learning and SHAP.</div>
</div>
""",
    unsafe_allow_html=True
)
# ============================================================
# 4. FORM INTRODUCTION
# ============================================================

st.markdown("## Customer Profile")
st.caption("Enter customer account, service, and billing information.")


# ============================================================
# 5. CUSTOMER FORM
# ============================================================

with st.form("customer_form"):


    # --------------------------------------------------------
    # CUSTOMER INFORMATION
    # --------------------------------------------------------

    st.subheader("Customer Information")

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


    st.divider()


    # --------------------------------------------------------
    # SERVICE INFORMATION
    # --------------------------------------------------------

    st.subheader("Service Information")

    col4, col5, col6 = st.columns(3)


    with col4:

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            [
                "No",
                "Yes",
                "No phone service"
            ]
        )


    with col5:

        internet_service = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

        online_security = st.selectbox(
            "Online Security",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )


    with col6:

        online_backup = st.selectbox(
            "Online Backup",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

        device_protection = st.selectbox(
            "Device Protection",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )


    st.divider()


    # --------------------------------------------------------
    # ADDITIONAL SERVICES
    # --------------------------------------------------------

    st.subheader("Additional Services")

    col7, col8, col9 = st.columns(3)


    with col7:

        tech_support = st.selectbox(
            "Tech Support",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )


    with col8:

        streaming_movies = st.selectbox(
            "Streaming Movies",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )


    with col9:

        paperless_billing = st.selectbox(
            "Paperless Billing",
            [
                "Yes",
                "No"
            ]
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


    st.divider()


    # --------------------------------------------------------
    # BILLING INFORMATION
    # --------------------------------------------------------

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


    st.write("")


    submit_button = st.form_submit_button(
        "Analyze Churn Risk"
    )


# ============================================================
# 6. VALIDATION AND PREDICTION
# ============================================================

if submit_button:


    # --------------------------------------------------------
    # INPUT VALIDATION
    # --------------------------------------------------------

    validation_errors = []


    if (
        phone_service == "No"
        and multiple_lines != "No phone service"
    ):

        validation_errors.append(
            "Multiple Lines must be 'No phone service' "
            "when Phone Service is No."
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

        for (
            feature_name,
            feature_value
        ) in internet_features.items():

            if (
                feature_value
                != "No internet service"
            ):

                validation_errors.append(
                    f"{feature_name} must be "
                    f"'No internet service' when "
                    f"Internet Service is No."
                )


    # --------------------------------------------------------
    # SHOW VALIDATION ERRORS
    # --------------------------------------------------------

    if validation_errors:

        st.error(
            "Please correct the customer information below."
        )

        for error in validation_errors:

            st.write(
                f"• {error}"
            )

        st.stop()


    # ========================================================
    # 7. PREPARE DATAFRAME
    # ========================================================

    customer_data = pd.DataFrame({

        "gender": [
            gender
        ],

        "SeniorCitizen": [
            1 if senior_citizen == "Yes" else 0
        ],

        "Partner": [
            partner
        ],

        "Dependents": [
            dependents
        ],

        "tenure": [
            tenure
        ],

        "PhoneService": [
            phone_service
        ],

        "MultipleLines": [
            multiple_lines
        ],

        "InternetService": [
            internet_service
        ],

        "OnlineSecurity": [
            online_security
        ],

        "OnlineBackup": [
            online_backup
        ],

        "DeviceProtection": [
            device_protection
        ],

        "TechSupport": [
            tech_support
        ],

        "StreamingTV": [
            streaming_tv
        ],

        "StreamingMovies": [
            streaming_movies
        ],

        "Contract": [
            contract
        ],

        "PaperlessBilling": [
            paperless_billing
        ],

        "PaymentMethod": [
            payment_method
        ],

        "MonthlyCharges": [
            monthly_charges
        ],

        "TotalCharges": [
            total_charges
        ]
    })


    # ========================================================
    # 8. CALL FLASK API
    # ========================================================

    API_URL = "http://127.0.0.1:5000/predict"


    try:

        with st.spinner(
            "Analyzing customer churn risk..."
        ):

            response = requests.post(

                API_URL,

                json=customer_data.iloc[0].to_dict(),

                timeout=10
            )


        # ====================================================
        # 9. SUCCESSFUL RESPONSE
        # ====================================================

        if response.status_code == 200:

            result = response.json()


            prediction = result[
                "prediction"
            ]


            churn_probability = result[
                "churn_probability"
            ]


            churn_factors = result[
                "factors_toward_churn"
            ]


            retention_factors = result[
                "factors_toward_retention"
            ]


            # ------------------------------------------------
            # RESULT HEADING AND PROBABILITY
            # ------------------------------------------------

            st.write("")
            st.markdown("## Prediction Result")
            st.caption("Model prediction and customer-specific risk analysis.")

            probability_percent = churn_probability * 100

            with st.container(border=True):
                st.metric(
                    label="Churn Probability",
                    value=f"{probability_percent:.2f}%"
                )


            # ------------------------------------------------
            # PREDICTION MESSAGE
            # ------------------------------------------------

            if prediction == "Churn":

                st.warning(
                    " ⚠️ Prediction: Customer is at risk of churn."
                )


            else:

                st.success(
                    " Prediction: Customer is not currently classified as churn risk."
                )


            # ------------------------------------------------
            # SHAP EXPLANATION HEADING
            # ------------------------------------------------

            st.write("")

            st.markdown("## Why This Prediction?")
            st.caption(
                "Customer-specific factors identified through SHAP analysis."
            )


            # ------------------------------------------------
            # SHAP FACTOR COLUMNS
            # ------------------------------------------------

            col_churn, col_retention = st.columns(2)


            # CHURN FACTORS

            with col_churn:

                with st.container(border=True):

                    st.markdown("###  Factors Toward Churn")


                    for factor in churn_factors:

                        st.write(f"• {factor}")

                        st.write("")


            # RETENTION FACTORS

            with col_retention:

                with st.container(border=True):

                    st.markdown("###  Factors Toward Retention")


                    for factor in retention_factors:

                        st.write(f"• {factor}")

                        st.write("")


        # ====================================================
        # 10. API ERROR
        # ====================================================

        else:

            try:

                error_message = response.json().get(
                    "message",
                    "Unknown API error"
                )


            except ValueError:

                error_message = (
                    "The prediction API returned "
                    "an invalid response."
                )


            st.error(
                f"Prediction failed: {error_message}"
            )


    # ========================================================
    # 11. CONNECTION ERROR
    # ========================================================

    except requests.exceptions.ConnectionError:

        st.error(
            "Cannot connect to the prediction API. "
            "Make sure the Flask backend is running."
        )


    # ========================================================
    # 12. OTHER REQUEST ERRORS
    # ========================================================

    except requests.exceptions.RequestException as e:

        st.error(
            f"API request failed: {e}"
        )


# ============================================================
# 13. FOOTER
# ============================================================

st.markdown(
    """
<div class="footer">
    Customer Churn Prediction System using  Machine Learning 
</div>
    """,
    unsafe_allow_html=True
)