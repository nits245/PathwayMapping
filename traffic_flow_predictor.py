import numpy as np
from tensorflow.keras.models import load_model
from data_processing import DataProcessing

class TrafficFlowPredictor:
    """
    Encapsulates SCATS data processing and Keras model prediction to provide
    time-of-day traffic flow counts efficiently.
    """
    def __init__(self, scats_filepath: str, model_path: str):
        # Load and preprocess SCATS data once
        self.dp = DataProcessing(scats_filepath)
        self.dp.process_scats_data()

        # Load Keras model once, supplying custom rmse if present
        self.model = load_model(
            model_path
        )

    def predict(self, scat_number: int, time_of_day: str) -> float:
        """
        Predict the raw 15-minute traffic count for a given SCATS sensor
        and time of day.

        Handles both single-input models (Conv1D, LSTM) and two-input models
        (GRU with temporal + location features).

        Args:
            scat_number: SCATS sensor ID
            time_of_day: String 'HH:MM'

        Returns:
            Raw count (float) for the 15-minute interval
        """
        # 1) Build the temporal sequence input (shape: [1, window, 1])
        seq = self.dp.get_sequence_by_time(scat_number, time_of_day)

        # 2) Determine if model expects multiple inputs
        if len(self.model.inputs) > 1:
            # Build the location input (shape: [1, 5])
            # Ensure DataProcessing implements get_location_vector
            loc_feats = np.array([
                self.dp.get_location_vector(scat_number)
            ], dtype=seq.dtype)

            # 3a) Two-input model (e.g., GRU)
            y_pred_scaled = self.model.predict([seq, loc_feats])
        else:
            # 3b) Single-input model (Conv1D, LSTM)
            y_pred_scaled = self.model.predict(seq)

        # Inverse scale to raw count
        raw_count = self.dp.scaler.inverse_transform(
            y_pred_scaled.reshape(-1, 1)
        )[0, 0]
        return float(raw_count)