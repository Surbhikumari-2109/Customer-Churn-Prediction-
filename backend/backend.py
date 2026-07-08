# 1. Import Required Libraries

from flask import Flask, request, jsonify
import pandas as pd
import joblib
from pathlib import Path

# 2. Create Flask Application

app = Flask(__name__)

# 3. Define Model Paths

BASE_DIR = Path(__file__).resolve().parent.parent

model_path = BASE_DIR / "models" / "churn_pipeline.pkl"
threshold_path = BASE_DIR / "models" / "churn_threshold.pkl"

# SHAP File Paths
shap_explainer_path = BASE_DIR / "models" / "shap_explainer.pkl"
shap_feature_names_path = BASE_DIR / "models" / "shap_feature_names.pkl"

# 4. Load Model Pipeline and Threshold

model = joblib.load(model_path)
threshold = joblib.load(threshold_path)

# Load SHAP Components
shap_explainer = joblib.load(shap_explainer_path)

shap_feature_names = joblib.load(
    shap_feature_names_path
)

# 5. Define Required Input Features

REQUIRED_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges"
]

# 6. Check Whether a SHAP Feature Is Active

def is_active_feature(feature_name, raw_customer):

    # Numeric features are always active
    if feature_name.startswith("num__"):
        return True

    clean_name = feature_name.replace("cat__", "")

    # Check whether one-hot encoded category is present
    for column in raw_customer.index:

        prefix = column + "_"

        if clean_name.startswith(prefix):

            category = clean_name[len(prefix):]

            return str(raw_customer[column]) == category

    return False

# 7. Convert Technical Feature Names to Readable Text

def make_readable_factor(feature_name, raw_customer):

    clean_name = (
        feature_name
        .replace("num__", "")
        .replace("cat__", "")
    )

    # Numeric features

    if clean_name == "tenure":

        tenure_value = raw_customer["tenure"]

        if tenure_value <= 12:
            return "Short customer tenure"

        elif tenure_value >= 48:
            return "Long customer tenure"

        else:
            return "Customer tenure"


    elif clean_name == "MonthlyCharges":

        monthly_value = raw_customer["MonthlyCharges"]

        if monthly_value >= 70:
            return "High monthly charges"

        elif monthly_value <= 35:
            return "Low monthly charges"

        else:
            return "Moderate monthly charges"
# Business-Friendly Categorical Explanations

    if clean_name == "MultipleLines_No":
        return "Single phone line"

    elif clean_name == "MultipleLines_Yes":
        return "Multiple phone lines"

    elif clean_name == "TechSupport_Yes":
        return "Tech support enabled"

    elif clean_name == "TechSupport_No":
        return "No tech support"

    elif clean_name == "OnlineSecurity_Yes":
        return "Online security enabled"

    elif clean_name == "OnlineSecurity_No":
        return "No online security"

    elif clean_name == "PaperlessBilling_Yes":
        return "Paperless billing enabled"

    elif clean_name == "PaperlessBilling_No":
        return "Paper billing"

    elif clean_name == "Dependents_Yes":
        return "Has dependents"

    elif clean_name == "Dependents_No":
        return "No dependents"

    elif clean_name == "Partner_Yes":
        return "Has a partner"

    elif clean_name == "Partner_No":
        return "No partner"

    elif clean_name == "TotalCharges":
        return "Total billing history"


    # Categorical features

    if "_" in clean_name:

        column, category = clean_name.split("_", 1)

        readable_columns = {
            "Contract": "Contract",
            "InternetService": "Internet service",
            "PaymentMethod": "Payment method",
            "TechSupport": "Tech support",
            "OnlineSecurity": "Online security",
            "OnlineBackup": "Online backup",
            "DeviceProtection": "Device protection",
            "PaperlessBilling": "Paperless billing",
            "Dependents": "Dependents",
            "Partner": "Partner",
            "MultipleLines": "Multiple lines",
            "StreamingTV": "Streaming TV",
            "StreamingMovies": "Streaming movies",
            "PhoneService": "Phone service",
            "gender": "Gender",
            "SeniorCitizen": "Senior citizen"
        }

        readable_column = readable_columns.get(
            column,
            column
        )

        return f"{readable_column}: {category}"

    return clean_name

# 8. Generate SHAP Explanation for One Customer

def generate_explanation(customer_df, top_n=3):

    # Get fitted preprocessor from saved pipeline
    preprocessor = model.named_steps["preprocessor"]

    # Transform raw customer data
    customer_transformed = preprocessor.transform(
        customer_df
    )

    # Calculate SHAP values
    customer_shap = shap_explainer(
        customer_transformed
    )

    shap_values = customer_shap.values[0]

    # Raw customer row
    raw_customer = customer_df.iloc[0]

    # Create explanation table
    explanation_df = pd.DataFrame({
        "Feature": shap_feature_names,
        "SHAP Value": shap_values
    })

    # Keep numeric and active categorical features
    explanation_df = explanation_df[
        explanation_df["Feature"].apply(
            lambda feature: is_active_feature(
                feature,
                raw_customer
            )
        )
    ]

    # Top factors toward churn
    churn_factors = (
        explanation_df[
            explanation_df["SHAP Value"] > 0
        ]
        .sort_values(
            "SHAP Value",
            ascending=False
        )
        .head(top_n)
    )

    # Top factors toward retention
    retention_factors = (
        explanation_df[
            explanation_df["SHAP Value"] < 0
        ]
        .sort_values(
            "SHAP Value",
            ascending=True
        )
        .head(top_n)
    )

    churn_reasons = [
        make_readable_factor(feature, raw_customer)
        for feature in churn_factors["Feature"]
    ]

    retention_reasons = [
        make_readable_factor(feature, raw_customer)
        for feature in retention_factors["Feature"]
    ]

    return churn_reasons, retention_reasons

# Create Health Check Endpoint

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "success",
        "message": "Customer Churn API is running"
    })

# Create Prediction Endpoint

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Receive JSON data from frontend
        customer_data = request.get_json()
        
        # Validate Request Data (Fixed Indentation)
        if not customer_data:
            return jsonify({
                "status": "error",
                "message": "No customer data provided."
            }), 400

        missing_features = [
            feature
            for feature in REQUIRED_FEATURES
            if feature not in customer_data
        ]

        if missing_features:
            return jsonify({
                "status": "error",
                "message": "Missing required features.",
                "missing_features": missing_features
            }), 400

        # Convert received data into one-row DataFrame
        customer_df = pd.DataFrame([customer_data])

        # Predict churn probability
        churn_probability = model.predict_proba(customer_df)[0][1]

        # Apply selected threshold
        prediction = (
            "Churn"
            if churn_probability >= threshold
            else "No Churn"
        )
        # Generate SHAP Explanation
        churn_reasons, retention_reasons = generate_explanation(
            customer_df,
            top_n=3
)

        # Return prediction response
        return jsonify({
            "prediction": prediction,
            "churn_probability": round(
                float(churn_probability), 4
            ),
            "threshold": float(threshold),
            "factors_toward_churn": churn_reasons,
            "factors_toward_retention": retention_reasons
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

# 4. Run Flask Application

if __name__ == "__main__":
    app.run(debug=True, port=5000)