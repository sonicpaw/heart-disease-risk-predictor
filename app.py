from flask import Flask, render_template, request, send_file
import os

from model_pipeline import predict_heart_risk
from explainable_ai import explain_prediction
from health_recommendations import generate_health_recommendations
from report_generator import generate_patient_report

app = Flask(__name__)

LAST_REPORT_PATH = None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    global LAST_REPORT_PATH

    # -------- Patient Name --------
    patient_name = request.form["patient_name"]

    # -------- Input Data (MATCHES ORIGINAL CSV) --------
    input_data = {
        "age": int(request.form["age"]),
        "gender": int(request.form["gender"]),
        "systolic_bp": int(request.form["systolic_bp"]),
        "diastolic_bp": int(request.form["diastolic_bp"]),
        "cholesterol": int(request.form["cholesterol"]),
        "diabetes": int(request.form["diabetes"]),
        "smoker": int(request.form["smoker"]),
        "physical_activity": int(request.form["physical_activity"]),
        "family_history": int(request.form["family_history"])
    }

    # -------- Prediction --------
    probability, risk_level_raw = predict_heart_risk(input_data)

    if risk_level_raw == "Low":
        risk_level = "Low Risk"
        risk_color = "success"
    elif risk_level_raw == "Medium":   # 🔥 FIXED (was Moderate)
        risk_level = "Moderate Risk"
        risk_color = "warning"
    else:
        risk_level = "High Risk"
        risk_color = "danger"

    # -------- Explainability (FIXED) --------
    explanation_dict = explain_prediction(input_data)

    explanation = [
        {"feature": k, "impact": v}
        for k, v in explanation_dict.items()
    ]

    contributors = [
        item["feature"].replace("_", " ").title()
        for item in explanation if item["impact"] > 0
    ]

    if not contributors:
        contributors = ["No major risk factors identified"]

    # -------- Health Recommendations --------
    recommendations = generate_health_recommendations(input_data)

    # -------- Generate Report --------
    LAST_REPORT_PATH = generate_patient_report(
        patient_name=patient_name,
        input_data=input_data
    )

    return render_template(
        "result.html",
        patient_name=patient_name,
        probability=probability,
        risk_level=risk_level,
        risk_color=risk_color,
        contributors=contributors,
        recommendations=recommendations
    )


@app.route("/download_report", methods=["POST"])
def download_report():
    global LAST_REPORT_PATH

    if LAST_REPORT_PATH and os.path.exists(LAST_REPORT_PATH):
        return send_file(LAST_REPORT_PATH, as_attachment=True)

    return "Report not found. Please generate a prediction first.", 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
