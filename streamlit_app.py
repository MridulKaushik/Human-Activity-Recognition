import csv
import streamlit as st
import requests
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import plotly.graph_objects as go

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("har.dashboard")

st.set_page_config(page_title="HAR Activity Classifier", layout="centered")

API_URL = st.secrets.get("API_URL", "http://127.0.0.1:8000/predict")
EXPLAIN_URL = API_URL.rsplit("/", 1)[0] + "/explain"
FEEDBACK_FILE = Path("feedback.csv")

st.title("Human Activity Recognition")
st.caption("Classifies activity from raw accelerometer + gyroscope sensor windows, using a Random Forest trained on hand-engineered signal-processing features.")

st.divider()

# ---------- Input method ----------
st.subheader("1. Provide a sensor window")
input_method = st.radio(
    "Choose input method",
    ["Upload a sample JSON file", "Paste JSON manually"],
    horizontal=True
)

sensor_data = None

if input_method == "Upload a sample JSON file":
    uploaded_file = st.file_uploader("Upload one of your sample_windows JSON files", type="json")
    if uploaded_file is not None:
        try:
            sensor_data = json.load(uploaded_file)
            st.success("File loaded successfully.")
        except json.JSONDecodeError:
            st.error("That file isn't valid JSON. Make sure you're uploading one activity's file, not the whole export.")

else:
    pasted = st.text_area("Paste the full JSON content here", height=200)
    if pasted.strip():
        try:
            sensor_data = json.loads(pasted)
            st.success("JSON parsed successfully.")
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")

st.divider()

# ---------- Prediction ----------
st.subheader("2. Run prediction")

if st.button("Predict Activity", type="primary", disabled=(sensor_data is None)):
    started = time.perf_counter()
    try:
        response = requests.post(API_URL, json=sensor_data, timeout=10)
        if response.status_code == 200:
            result = response.json()
            st.session_state["prediction_result"] = result
            logger.info(
                "Prediction activity=%s confidence=%.4f latency_ms=%.2f",
                result["activity"],
                result["confidence"],
                (time.perf_counter() - started) * 1000,
            )

            st.success(f"Predicted activity: **{result['activity']}**  (confidence: {result['confidence']:.1%})")

            # Probability bar chart, sorted descending
            probs = result['all_probabilities']
            sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            labels = [k for k, v in sorted_items]
            values = [v for k, v in sorted_items]

            fig = go.Figure(go.Bar(
                x=values,
                y=labels,
                orientation='h',
                marker_color=['#2E86AB' if l == result['activity'] else '#B0BEC5' for l in labels],
                text=[f"{v:.1%}" for v in values],
                textposition='auto'
            ))
            fig.update_layout(
                title="Prediction confidence by class",
                xaxis_title="Probability",
                xaxis=dict(range=[0, 1]),
                height=350,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            logger.warning("Prediction API error status=%s body=%s", response.status_code, response.text)
            st.error(f"API returned an error ({response.status_code}): {response.text}")

    except requests.exceptions.ConnectionError:
        logger.exception("Prediction API connection failed")
        st.error("Couldn't reach the API. Make sure `uvicorn app:app --reload` is running in a separate terminal on port 8000.")
    except (requests.RequestException, ValueError) as error:
        logger.exception("Prediction request failed: %s", error)
        st.error(f"Prediction failed: {error}")

result = st.session_state.get("prediction_result")
if result is not None:
    st.subheader("Why did the model make this prediction?")
    if st.button("Generate SHAP explanation"):
        try:
            response = requests.post(EXPLAIN_URL, json=sensor_data, timeout=30)
            response.raise_for_status()
            explanation = response.json()
            shap_rows = explanation["features"]
            shap_rows.reverse()
            shap_fig = go.Figure(
                go.Bar(
                    x=[row["shap_value"] for row in shap_rows],
                    y=[row["feature"] for row in shap_rows],
                    orientation="h",
                    marker_color=[
                        "#2E86AB" if row["shap_value"] >= 0 else "#D1495B"
                        for row in shap_rows
                    ],
                    customdata=[[row["feature_value"]] for row in shap_rows],
                    hovertemplate=(
                        "%{y}<br>SHAP contribution: %{x:.4f}"
                        "<br>Feature value: %{customdata[0]:.4f}<extra></extra>"
                    ),
                )
            )
            shap_fig.update_layout(
                title=f"Top SHAP contributions for {explanation['activity']}",
                xaxis_title="Contribution to predicted class",
                yaxis_title="Feature",
                height=500,
                margin=dict(l=10, r=10, t=55, b=10),
            )
            st.plotly_chart(shap_fig, use_container_width=True)
            logger.info("SHAP explanation generated for activity=%s", explanation["activity"])
        except (requests.RequestException, ValueError, KeyError) as error:
            logger.exception("SHAP explanation failed: %s", error)
            st.error(f"Could not generate SHAP explanation: {error}")

    st.subheader("Was this prediction useful?")
    st.caption("Your feedback is stored locally in feedback.csv for future model improvement.")
    feedback_columns = st.columns(2)

    def save_feedback(value: str) -> None:
        """Append a user rating with the prediction context to a local CSV."""
        file_exists = FEEDBACK_FILE.exists()
        with FEEDBACK_FILE.open("a", newline="", encoding="utf-8") as feedback_file:
            writer = csv.DictWriter(
                feedback_file,
                fieldnames=["timestamp", "activity", "confidence", "feedback"],
            )
            if not file_exists:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "activity": result["activity"],
                    "confidence": result["confidence"],
                    "feedback": value,
                }
            )
        logger.info("Feedback recorded activity=%s feedback=%s", result["activity"], value)
        st.session_state["feedback_recorded"] = value

    with feedback_columns[0]:
        if st.button("Thumbs Up", key="feedback_up"):
            save_feedback("up")
    with feedback_columns[1]:
        if st.button("Thumbs Down", key="feedback_down"):
            save_feedback("down")
    if st.session_state.get("feedback_recorded"):
        st.success("Thanks. Your feedback was recorded.")

st.divider()

# ---------- Model comparison context ----------
st.subheader("Model comparison")
st.markdown("""
This dashboard serves the Random Forest model, which outperformed a CNN and LSTM
trained directly on raw signals:

| Model | Accuracy | Sitting F1 | Standing F1 |
|---|---|---|---|
| **Random Forest (engineered features)** | **0.95** | **0.93** | **0.94** |
| CNN (raw signals) | 0.94 | 0.85 | 0.87 |
| LSTM, pooled (raw signals) | 0.92 | 0.81 | 0.85 |

Hand-engineered gravity/orientation features gave the classical model a clear edge on
static postures (sitting, standing), where motion-based deep learning features struggled.
""")