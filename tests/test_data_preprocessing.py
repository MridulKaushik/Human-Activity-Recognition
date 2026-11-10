import numpy as np

from src.data_preprocessing import build_feature_row, validate_sensor_window


def sensor_window() -> dict[str, list[float]]:
    values = np.linspace(-1, 1, 128).tolist()
    return {
        channel: values
        for channel in (
            "body_acc_x",
            "body_acc_y",
            "body_acc_z",
            "body_gyro_x",
            "body_gyro_y",
            "body_gyro_z",
            "total_acc_x",
            "total_acc_y",
            "total_acc_z",
        )
    }


def test_feature_row_matches_persisted_feature_count() -> None:
    row = build_feature_row(sensor_window())
    assert len(row) == 63


def test_sensor_window_validation_rejects_wrong_length() -> None:
    data = sensor_window()
    data["body_acc_x"] = [0.0] * 127
    try:
        validate_sensor_window(data)
    except ValueError as error:
        assert "128" in str(error)
    else:
        raise AssertionError("Expected wrong-length input to be rejected")
