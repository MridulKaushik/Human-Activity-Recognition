from __future__ import annotations

from pathlib import Path
from typing import Any

import pickle


def save_model_artifacts(
    model: Any, feature_columns: list[str], output_dir: Path
) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "har_rf_model.pkl").open("wb") as model_file:
        pickle.dump(model, model_file)
    with (output_dir / "feature_columns.pkl").open("wb") as columns_file:
        pickle.dump(feature_columns, columns_file)
