"""
Model training module with experiment tracking and monitoring.

Implements CNN-LSTM architecture with TensorBoard and W&B integration
for comprehensive experiment tracking and model performance monitoring.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from tensorflow.keras.layers import (
    LSTM,
    Activation,
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Input,
    LayerNormalization,
)
from tensorflow.keras.models import Sequential

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from src.config_loader import get_config
from src.logger import setup_logger

keras.saving.get_custom_objects().clear()


@keras.saving.register_keras_serializable(package="Custom")
class DirectionalLogCoshLoss(keras.losses.Loss):
    def __init__(self, penalty_weight=2.0, name="directional_log_cosh", **kwargs):
        super().__init__(name=name, **kwargs)
        self.penalty_weight = float(penalty_weight)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        error = y_pred - y_true
        log_cosh = tf.math.log(tf.math.cosh(error + 1e-12))

        # Penalize incorrect direction
        same_direction = tf.cast(
            tf.math.sign(y_true) == tf.math.sign(y_pred), tf.float32
        )
        penalty = same_direction + (1.0 - same_direction) * self.penalty_weight

        return tf.reduce_mean(log_cosh * penalty)

    def get_config(self):
        config = super().get_config()
        config.update({"penalty_weight": self.penalty_weight})
        return config


class ModelTrainer:
    """Handles model training with monitoring and experiment tracking."""

    def __init__(self, config_path: str = None, use_wandb: bool = None, **kwargs):

        super().__init__(**kwargs)

        self.config = get_config(config_path)
        self.logger = setup_logger("model_training")

        self.data_dir = Path(self.config.data["data_dir"])
        self.model_dir = Path(self.config.deployment["model_dir"])
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.config = get_config(config_path)
        # Monitoring setup
        self.use_tensorboard = self.config.monitoring["tensorboard"]["enabled"]
        self.use_wandb = (
            use_wandb
            if use_wandb is not None
            else self.config.monitoring["wandb"]["enabled"]
        )

        if self.use_wandb:
            try:
                import wandb

                self.wandb = wandb
                self._init_wandb()
            except ImportError:
                self.logger.warning(
                    "W&B not installed. Install with: pip install wandb"
                )
                self.use_wandb = False

        # Training metadata
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"fx_experiment_{self.timestamp}"

    def _init_wandb(self):
        """Initialize Weights & Biases tracking."""
        wandb_config = self.config.monitoring["wandb"]

        self.wandb.init(
            project=wandb_config["project"],
            entity=wandb_config["entity"],
            name=self.experiment_name,
            tags=wandb_config["tags"],
            config={
                "model": self.config.model,
                "training": self.config.training,
                "preprocessing": self.config.preprocessing,
            },
        )
        self.logger.info("W&B tracking initialized")

    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Build CNN-LSTM hybrid model for time series prediction.

        Args:
            input_shape: (window_size, num_features)

        Returns:
            Compiled Keras model
        """
        model_config = self.config.model

        model = Sequential(
            [
                Input(shape=input_shape),
                # Spatial dropout for time series
                tf.keras.layers.SpatialDropout1D(model_config["spatial_dropout"]),
                BatchNormalization(),
                # CNN layer for local pattern extraction
                Conv1D(
                    filters=model_config["conv_filters"],
                    kernel_size=model_config["conv_kernel_size"],
                    padding="causal",
                ),
                BatchNormalization(),
                Activation("swish"),
                # LSTM layers for sequential dependencies
                LSTM(
                    model_config["lstm_units"][0],
                    return_sequences=True,
                    recurrent_dropout=0.1,
                ),
                LayerNormalization(),
                Dropout(model_config["dropout_rate_lstm_1"]),
                LSTM(
                    model_config["lstm_units"][1],
                    return_sequences=False,
                    recurrent_dropout=0.1,
                ),
                LayerNormalization(),
                Dropout(model_config["dropout_rate_lstm_2"]),
                # Dense layers
                Dense(
                    model_config["dense_units"],
                    activation="swish",
                ),
                BatchNormalization(),
                Dropout(model_config["dropout_rate_dense"]),
                Dense(1, dtype="float32"),
            ]
        )

        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=model_config["learning_rate"],
            weight_decay=1e-4,
            amsgrad=True,
        )

        model.compile(
            optimizer=optimizer,
            loss=DirectionalLogCoshLoss(
                penalty_weight=model_config["directional_penalty"]
            ),
            metrics=["mae"],
        )

        return model

    def create_sequences(
        self, data: np.ndarray, target: np.ndarray, window_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding window sequences for time series.

        Args:
            data: Feature array
            target: Target array
            window_size: Length of input sequence

        Returns:
            Tuple of (X sequences, y targets)
        """
        X, y = [], []

        for i in range(len(data) - window_size):
            X.append(data[i : i + window_size])
            y.append(target[i + window_size])

        return np.array(X), np.array(y)

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and prepare data for training.

        Returns:
            Tuple of (features, targets)
        """
        data_path = self.data_dir / self.config.data["processed_file"]

        if not data_path.exists():
            raise FileNotFoundError(
                f"Processed data not found: {data_path}. "
                "Run data_preprocessing.py first."
            )

        df = pd.read_csv(data_path, index_col=0)
        df.index = pd.to_datetime(df.index)

        # Clean data
        df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()

        # Scale target for better training dynamics
        y_raw = (
            df["Target"].values.astype("float32") * self.config.training["target_scale"]
        )
        X_raw = df.drop(columns=["Target"]).values.astype("float32")

        self.logger.info(
            f"Loaded data: {X_raw.shape[0]} samples, {X_raw.shape[1]} features"
        )

        return X_raw, y_raw

    def get_callbacks(self, fold: int) -> list:
        """
        Create callbacks for model training.

        Args:
            fold: Current fold number

        Returns:
            List of Keras callbacks
        """
        callbacks = []

        # Early stopping
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=self.config.model["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        )
        callbacks.append(early_stop)

        # Model checkpoint
        checkpoint_path = self.model_dir / f"{self.experiment_name}_fold{fold}.keras"
        checkpoint = ModelCheckpoint(
            str(checkpoint_path), monitor="val_loss", save_best_only=True, verbose=1
        )
        callbacks.append(checkpoint)

        # Learning rate reduction
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
        )
        callbacks.append(reduce_lr)

        # TensorBoard
        if self.use_tensorboard:
            tb_log_dir = (
                Path(self.config.monitoring["tensorboard"]["log_dir"])
                / self.experiment_name
                / f"fold_{fold}"
            )
            tb_log_dir.mkdir(parents=True, exist_ok=True)

            tensorboard = TensorBoard(
                log_dir=str(tb_log_dir),
                histogram_freq=1,
                update_freq=self.config.monitoring["tensorboard"]["update_freq"],
            )
            callbacks.append(tensorboard)
            self.logger.info(f"TensorBoard logging to: {tb_log_dir}")

        # W&B callback
        if self.use_wandb:
            from wandb.integration.keras import WandbCallback

            wandb_callback = WandbCallback(
                save_model=False, log_weights=True, log_gradients=True
            )
            callbacks.append(wandb_callback)

        return callbacks

    def train_model(self):
        """Main training pipeline with cross-validation."""
        self.logger.info("Starting model training pipeline")

        # Load data
        X_raw, y_raw = self.load_data()

        # Create sequences
        window_size = self.config.model["window_size"]
        X, y = self.create_sequences(X_raw, y_raw, window_size)
        self.logger.info(f"Created {X.shape[0]} sequences of {window_size} timesteps")

        # Time series cross-validation
        n_splits = self.config.training["cv_splits"]
        tscv = TimeSeriesSplit(n_splits=n_splits, max_train_size=1250)

        fold_results = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            self.logger.info(f"\n{'=' * 50}")
            self.logger.info(f"Training Fold {fold}/{n_splits}")
            self.logger.info(f"{'=' * 50}")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Scale features per fold to prevent data leakage
            scaler = RobustScaler()
            samples, timesteps, features = X_train.shape

            X_train_scaled = scaler.fit_transform(
                X_train.reshape(-1, features)
            ).reshape(samples, timesteps, features)

            X_test_scaled = scaler.transform(X_test.reshape(-1, features)).reshape(
                X_test.shape
            )

            # Build model
            model = self.build_model((window_size, features))

            # Train model
            history = model.fit(
                X_train_scaled,
                y_train,
                epochs=self.config.model["epochs"],
                batch_size=self.config.model["batch_size"],
                validation_data=(X_test_scaled, y_test),
                callbacks=self.get_callbacks(fold),
                verbose=1,
            )

            # Evaluate
            predictions = model.predict(X_test_scaled, verbose=0).flatten()

            # Calculate metrics
            directional_accuracy = (
                np.mean(np.sign(predictions) == np.sign(y_test)) * 100
            )

            mae = np.mean(np.abs(predictions - y_test))

            self.logger.info(
                f"Fold {fold} Results - "
                f"Directional Accuracy: {directional_accuracy:.2f}%, "
                f"MAE: {mae:.4f}"
            )

            fold_result = {
                "fold": fold,
                "directional_accuracy": directional_accuracy,
                "mae": mae,
                "train_samples": len(train_idx),
                "test_samples": len(test_idx),
            }
            fold_results.append(fold_result)

            # Log to W&B
            if self.use_wandb:
                self.wandb.log(
                    {
                        f"fold_{fold}_directional_accuracy": directional_accuracy,
                        f"fold_{fold}_mae": mae,
                    }
                )

            # Save final fold model separately
            if fold == n_splits:
                final_model_path = (
                    self.model_dir / f"{self.experiment_name}_final.keras"
                )
                model.save(final_model_path)
                self.logger.info(f"Final model saved: {final_model_path}")

        # Save experiment summary
        summary = {
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "config": {"model": self.config.model, "training": self.config.training},
            "fold_results": fold_results,
            "average_directional_accuracy": np.mean(
                [r["directional_accuracy"] for r in fold_results]
            ),
            "average_mae": np.mean([r["mae"] for r in fold_results]),
        }

        summary_path = self.model_dir / f"{self.experiment_name}_summary.json"

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        self.logger.info(f"\nExperiment summary saved: {summary_path}")
        self.logger.info(
            f"\nOverall Performance:\n"
            f"  Average Directional Accuracy: "
            f"{summary['average_directional_accuracy']:.2f}%\n"
            f"  Average MAE: {summary['average_mae']:.4f}"
        )

        if self.use_wandb:
            self.wandb.log(
                {
                    "final_avg_directional_accuracy": summary[
                        "average_directional_accuracy"
                    ],
                    "final_avg_mae": summary["average_mae"],
                }
            )
            self.wandb.finish()

        return summary


def main():
    """Entry point for model training."""
    trainer = ModelTrainer()
    summary = trainer.train_model()
    print("\nTraining complete!")
    print(f"Experiment: {summary['experiment_name']}")
    print(
        f"Average Directional Accuracy: {summary['average_directional_accuracy']:.2f}%"
    )


if __name__ == "__main__":
    main()
