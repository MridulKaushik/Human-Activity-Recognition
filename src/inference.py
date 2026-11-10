import pickle
from pathlib import Path

import numpy as np
import shap

from .data_preprocessing import build_feature_frame, validate_sensor_window

ACTIVITY_LABELS = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}


class HARInference:

    def __init__(self, model_dir=None):
        root = Path(__file__).resolve().parents[1]
        artifact_dir = model_dir or root / "model"
        with (artifact_dir / "har_rf_model.pkl").open("rb") as model_file:
            self.model = pickle.load(model_file)
        with (artifact_dir / "feature_columns.pkl").open("rb") as columns_file:
            self.feature_columns = pickle.load(columns_file)

    def features(self, data):
        validate_sensor_window(data)
        return build_feature_frame(data, self.feature_columns)

    def predict(self, data):
        features = self.features(data)
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        classes = getattr(
            self.model, "classes_", range(1, len(probabilities) + 1)
        )
        return {
            "activity": ACTIVITY_LABELS[int(prediction)],
            "confidence": float(probabilities.max()),
            "all_probabilities": {
                ACTIVITY_LABELS[int(label)]: float(probability)
                for label, probability in zip(classes, probabilities)
            },
        }

    def explain(self, data, top_n=15):
        features = self.features(data)
        prediction = self.model.predict(features)[0]
        classes = list(getattr(self.model, "classes_", range(1, 7)))
        class_index = classes.index(prediction)
        shap_values = shap.TreeExplainer(self.model)(features).values
        values = np.asarray(shap_values)
        if values.ndim == 3:
            values = values[0, :, class_index]
        else:
            values = values[0]
        ranked = np.argsort(np.abs(values))[::-1][:top_n]
        return {
            "activity": ACTIVITY_LABELS[int(prediction)],
            "features": [
                {
                    "feature": self.feature_columns[index],
                    "shap_value": float(values[index]),
                    "feature_value": float(features.iloc[0, index]),
                }
                for index in ranked
            ],
        }
