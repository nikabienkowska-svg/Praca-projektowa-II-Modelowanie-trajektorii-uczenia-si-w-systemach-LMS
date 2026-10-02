"""Test sprawdzający spójność całego pipeline'u: mock data -> features -> baselines -> LSTM.
Uruchom po aktywacji środowiska wirtualnego:
    python tests/test_pipeline.py
"""

import os
import shutil

from src.data.download_oulad import generate_mock_oulad_sample
from src.data.loader import load_raw_tables
from src.features.sequence_builder import build_student_sequences
from src.models.baselines import train_baseline_models
from src.models.lstm_model import StudentDropoutLSTM
from src.models.train import train_lstm_pipeline
from src.interpretability.motivation_patterns import extract_inactivity_crisis_patterns


def test_entire_lms_pipeline():
    test_raw_dir = "data/test_raw"
    try:
        # 1. Mock data generation
        generate_mock_oulad_sample(output_dir=test_raw_dir, num_students=50)

        # 2. Data Loader
        data = load_raw_tables(raw_dir=test_raw_dir)
        assert "studentVle" in data
        assert len(data["studentInfo"]) == 50

        # 3. Sequence building
        X, y, student_ids, feature_names = build_student_sequences(
            student_vle=data["studentVle"],
            vle=data["vle"],
            student_reg=data["studentRegistration"],
            observation_window_days=30,
            prediction_horizon_days=7,
        )
        assert X.shape[0] == 50
        assert X.shape[1] == 30
        assert X.shape[2] == len(feature_names)
        assert len(y) == 50

        # 4. Baselines
        X_np = X.numpy()
        y_np = y.numpy()
        split = 35
        baseline_results = train_baseline_models(
            X_train_seq=X_np[:split],
            y_train=y_np[:split],
            X_val_seq=X_np[split:],
            y_val=y_np[split:],
            feature_names=feature_names,
        )
        assert "LogisticRegression" in baseline_results
        assert "RandomForest" in baseline_results

        # 5. Deep Learning LSTM training
        model, history = train_lstm_pipeline(
            X_train=X[:split],
            y_train=y[:split],
            X_val=X[split:],
            y_val=y[split:],
            hidden_dim=16,
            num_layers=1,
            batch_size=8,
            epochs=2,
        )
        assert isinstance(model, StudentDropoutLSTM)
        assert len(history["train_loss"]) == 2

        # 6. Motivation crisis patterns
        crisis_df = extract_inactivity_crisis_patterns(
            student_vle=data["studentVle"],
            student_reg=data["studentRegistration"],
            vle=data["vle"],
            pre_dropout_window_days=7,
        )
        assert isinstance(crisis_df, object)

        print("\n✅ Wszystkie testy pipeline'u przeszły pomyślnie!")
    finally:
        if os.path.exists(test_raw_dir):
            shutil.rmtree(test_raw_dir)


if __name__ == "__main__":
    test_entire_lms_pipeline()
