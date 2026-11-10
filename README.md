# Human Activity Recognition

This project classifies six human activities from raw smartphone inertial sensor
windows using a persisted Random Forest model and hand-engineered signal
processing features.

The supported activities are:

- Walking
- Walking upstairs
- Walking downstairs
- Sitting
- Standing
- Laying

## Dataset format

The model expects a 2.56-second sensor window sampled at 50 Hz. Each input
window must contain exactly 128 finite numeric values for each of these nine
channels:

```text
body_acc_x, body_acc_y, body_acc_z
body_gyro_x, body_gyro_y, body_gyro_z
total_acc_x, total_acc_y, total_acc_z
```

The channel names and window format follow the raw inertial-signal layout of
the [UCI Human Activity Recognition dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones).
The dataset itself is not included in this repository.

## Feature extraction

The application derives 63 features from each sensor window:

- Time-domain features: mean, standard deviation, RMS, minimum, maximum, and
  zero-crossing count
- Frequency-domain features: dominant frequency, spectral energy, and spectral
  entropy using Welch's power spectral density
- Gravity/orientation features: mean and standard deviation for the three
  total-acceleration axes
- Correlation features between the body-acceleration axes

The persisted feature-column list is used during inference so that features are
passed to the model in the same order used when the model artifact was created.

## Application

### FastAPI service

[`app.py`](app.py) exposes the trained Random Forest through these endpoints:

- `GET /health` - confirms that the service and model are available
- `POST /predict` - returns the predicted activity, confidence, and class
  probabilities
- `POST /explain` - returns the top SHAP feature contributions for the
  prediction

Requests are validated with Pydantic. Invalid or incorrectly sized sensor
windows are rejected.

### Streamlit dashboard

[`streamlit_app.py`](streamlit_app.py) provides a dashboard that:

- Accepts a sensor-window JSON file or pasted JSON
- Calls the FastAPI prediction endpoint
- Displays the predicted activity and probability chart
- Displays a SHAP explanation
- Records thumbs-up/thumbs-down feedback locally in `feedback.csv`

The dashboard uses `API_URL` from Streamlit secrets when configured. Otherwise,
it uses `http://127.0.0.1:8000/predict`.

## Running locally with a virtual environment

From the project directory, create and activate a Python virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\activate
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, allow local scripts once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Start the FastAPI service in the first terminal:

```powershell
python -m uvicorn app:app --reload
```

In a second terminal, activate the environment and start the dashboard:

```powershell
cd "<File-Path>\Human-activity-recognition"
.\venv\Scripts\activate
python -m streamlit run streamlit_app.py
```

Open the Streamlit URL shown in the terminal, usually
`http://localhost:8501`.

The API is available at `http://127.0.0.1:8000`. Interactive API
documentation is available at `http://127.0.0.1:8000/docs`.

## Docker

The Docker image runs the FastAPI service on port 8000:

```powershell
docker build -t har-api .
docker run --rm -p 8000:8000 har-api
```

## Testing and quality checks

Run the focused tests with:

```powershell
python -m pytest
```

The CI workflow also checks Black formatting, Flake8, MyPy, and test coverage
for `app.py`, `src/`, and `tests/`.

## Repository structure

```text
app.py                  FastAPI prediction service
streamlit_app.py        Streamlit dashboard
src/
  data_preprocessing.py  Feature extraction and sensor validation
  inference.py           Model loading, prediction, and SHAP explanations
  train_model.py        Helper for saving model artifacts
tests/
  test_data_preprocessing.py
                         Focused feature and validation tests
model/
  har_rf_model.pkl       Persisted Random Forest model
  feature_columns.pkl    Persisted feature order
assets/                  Dashboard and analysis images
Dockerfile               FastAPI container configuration
requirements.txt         Python dependencies
```

## Current scope

This repository contains the trained-model inference application. It does not
include a complete training pipeline, CNN/LSTM implementation, or a
reproducible cross-validation training script. The model artifacts in `model/`
are required to run the API.
