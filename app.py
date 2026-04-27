import pandas as pd
from flask import Flask, request, render_template, send_file, flash
import joblib
import os
from werkzeug.utils import secure_filename

# Flask app
app = Flask(__name__)
app.secret_key = "phishing_secret_key"

# Folders
UPLOAD_FOLDER = "/tmp/user_uploads"
OUTPUT_FOLDER = "/tmp/Prediction_result"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load saved model + scaler + features
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
feature_names = joblib.load(os.path.join(BASE_DIR, "feature_names.pkl"))

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Predict route (CSV upload)
@app.route("/predict", methods=["POST"])
def predict():

    # Check file exists
    if "file" not in request.files:
        flash("No file uploaded!")
        return render_template("index.html")

    file = request.files["file"]

    # Check file selected
    if file.filename == "":
        flash("No file selected!")
        return render_template("index.html")

    # Save uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    # Read CSV
    df = pd.read_csv(file_path)

    # Check missing columns
    missing_cols = set(feature_names) - set(df.columns)
    if missing_cols:
        flash(f"Missing columns: {sorted(list(missing_cols))}")
        return render_template("index.html")

    # Select features in correct order
    X_new = df[feature_names]

    # Scale data
    X_new_scaled = scaler.transform(X_new)

    # Predict
    predictions = model.predict(X_new_scaled)

    # Add results
    df["prediction"] = predictions
    df["prediction_label"] = df["prediction"].map({
        -1: "Phishing",
         0: "Suspicious",
         1: "Legitimate"
    })

    # Save output csv
    output_path = os.path.join(OUTPUT_FOLDER, "prediction_result.csv")
    df.to_csv(output_path, index=False)
    return send_file(output_path, as_attachment=True)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
