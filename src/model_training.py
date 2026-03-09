import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, LeakyReLU, BatchNormalization, Conv1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# 1. HARDWARE SETTINGS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def build_lstm_model(input_shape):
    """Build LSTM model with proper regression output and stronger regularization."""
    model = Sequential([
        Input(shape=input_shape),
        BatchNormalization(),
        Conv1D(filters=16, kernel_size=3, activation='relu', padding='causal'),
        Dropout(0.2),
        
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        
        LSTM(32, return_sequences=False, recurrent_regularizer=l2(1e-5)),
        Dropout(0.3),
        
        Dense(16, activation='relu'),
        LeakyReLU(alpha=0.1),
        Dropout(0.3),
        
        Dense(1, activation='linear')  # Explicit linear activation for regression
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])
    return model


def calculate_metrics(y_true, y_pred, name=""):
    """Calculate comprehensive regression metrics."""
    # Correlation (most important for directional prediction)
    if np.std(y_pred) > 1e-8 and np.std(y_true) > 1e-8:
        correlation, _ = pearsonr(y_pred, y_true)
    else:
        correlation = 0.0
    
    # Mean Absolute Error
    mae = np.mean(np.abs(y_pred - y_true))
    
    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    
    # Directional accuracy (only meaningful if target has both signs)
    unique_signs = len(np.unique(np.sign(y_true)))
    if unique_signs > 1:
        directional_acc = np.mean(np.sign(y_pred) == np.sign(y_true)) * 100
    else:
        directional_acc = None
    
    # Prediction quality checks
    pred_std = np.std(y_pred)
    true_std = np.std(y_true)
    
    print(f"\n{name} Metrics:")
    print(f"  Correlation: {correlation:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    if directional_acc is not None:
        print(f"  Directional Acc: {directional_acc:.2f}%")
    print(f"  Pred Std: {pred_std:.4f} | True Std: {true_std:.4f}")
    
    return {
        "correlation": correlation,
        "mae": mae,
        "rmse": rmse,
        "directional_acc": directional_acc,
        "pred_std": pred_std,
    }


def run_training_pipeline():
    # Fix path resolution to work regardless of execution method
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "../data/data.csv")
    
    try:
        df = pd.read_csv(data_path, index_col=0)
        print(f"✓ Loaded data from: {data_path}")
        print(f"  Shape: {df.shape}")
    except FileNotFoundError:
        print(f"✗ CSV not found at: {data_path}")
        print("  Please ensure the file exists at the correct path.")
        return
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return

    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna().astype('float32')
    
    if df.empty:
        print("✗ DataFrame is empty after cleaning.")
        return

    # Analyze target distribution
    print("\n" + "=" * 60)
    print("TARGET DISTRIBUTION ANALYSIS")
    print("=" * 60)
    target_raw = df['Target']
    print(f"Target stats:\n{target_raw.describe()}")
    print(f"\nPositive: {(target_raw > 0).sum() / len(target_raw) * 100:.2f}%")
    print(f"Negative: {(target_raw < 0).sum() / len(target_raw) * 100:.2f}%")
    print(f"Zero: {(target_raw == 0).sum() / len(target_raw) * 100:.2f}%")
    
    # Check if target needs transformation
    if (target_raw >= 0).all():
        print("\n⚠ WARNING: Target is all non-negative!")
        print("  Consider using returns or differences that can be negative")
        print("  for meaningful directional prediction.")

    # Prepare features and target (keep original scale for interpretability)
    target = df['Target'].values.astype(np.float32)
    features = df.drop(columns=['Target']).values.astype(np.float32)

    window_size = 5
    X, y = [], []
    for i in range(len(features) - window_size):
        X.append(features[i : i + window_size])
        y.append(target[i + window_size])
    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    
    print(f"\nSequence data created:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")

    # Use TimeSeriesSplit for walk-forward validation
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)

    results_table = []
    all_test_correlations = []

    print("\n" + "=" * 60)
    print("STARTING WALK-FORWARD VALIDATION")
    print("=" * 60)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        fold_num = fold + 1
        print(f"\n--- FOLD {fold_num} ---")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = RobustScaler()
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        X_test_flat = X_test.reshape(-1, X_test.shape[-1])

        X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
        X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)

        # Build and train model
        model = build_lstm_model((X_train_scaled.shape[1], X_train_scaled.shape[2]))

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0
        )
        early_stop = EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True, verbose=0
        )

        history = model.fit(
            X_train_scaled, y_train,
            epochs=100,
            batch_size=64,
            validation_data=(X_test_scaled, y_test),
            callbacks=[reduce_lr, early_stop],
            verbose=0
        )

        # Predictions
        y_train_pred = model.predict(X_train_scaled, verbose=0).flatten()
        y_test_pred = model.predict(X_test_scaled, verbose=0).flatten()

        # Calculate metrics
        train_metrics = calculate_metrics(y_train, y_train_pred, f"Fold {fold_num} - Train")
        test_metrics = calculate_metrics(y_test, y_test_pred, f"Fold {fold_num} - Test")
        
        # Use correlation as primary metric for WFE
        train_corr = train_metrics["correlation"]
        test_corr = test_metrics["correlation"]
        wfe = test_corr / train_corr if abs(train_corr) > 1e-8 else 0

        # Store results
        fold_data = {
            "Fold": fold_num,
            "Train Samples": len(train_idx),
            "Test Samples": len(test_idx),
            "Train Corr": f"{train_corr:.4f}",
            "Test Corr": f"{test_corr:.4f}",
            "Test MAE": f"{test_metrics['mae']:.4f}",
            "WFE": round(wfe, 4),
        }
        results_table.append(fold_data)
        all_test_correlations.append(test_corr)

        print(f"\n✓ Fold {fold_num} complete | Test Corr: {test_corr:.4f} | WFE: {wfe:.4f}")
        
        # Visualization
        plt.figure(figsize=(15, 6))
        zoom_range = min(100, len(y_test))
        plt.plot(y_test[-zoom_range:], label='Actual', color='black', alpha=0.5, linewidth=1)
        plt.plot(y_test_pred[-zoom_range:], label='Prediction', color='red', linewidth=1.5)
        plt.axhline(0, color='blue', linestyle='--', alpha=0.3)
        plt.title(f"Fold {fold_num} | Corr: {test_corr:.4f} | MAE: {test_metrics['mae']:.4f}")
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(f'/workspace/fold_{fold_num}_predictions.png', dpi=100)
        plt.close()
        print(f"  Plot saved to: /workspace/fold_{fold_num}_predictions.png")

    # Final Report
    print("\n" + "=" * 95)
    print(
        f"{'Fold':<6} | {'Train Samples':<14} | {'Test Samples':<13} | "
        f"{'Train Corr':<10} | {'Test Corr':<10} | {'Test MAE':<10} | {'WFE':<6}"
    )
    print("-" * 95)
    for res in results_table:
        print(
            f"{res['Fold']:<6} | {res['Train Samples']:<14} | {res['Test Samples']:<13} | "
            f"{res['Train Corr']:<10} | {res['Test Corr']:<10} | {res['Test MAE']:<10} | {res['WFE']:<6}"
        )

    # Overall statistics
    mean_corr = np.mean(all_test_correlations)
    std_corr = np.std(all_test_correlations)
    overall_wfe = np.mean([r["WFE"] for r in results_table])
    
    print("-" * 95)
    print(f"{'SUMMARY':<67}")
    print(f"  Mean Test Correlation: {mean_corr:.4f} ± {std_corr:.4f}")
    print(f"  Overall WFE: {overall_wfe:.4f}")
    
    # Interpretation
    print("\nINTERPRETATION:")
    if mean_corr > 0.1:
        print("  ✓ Model shows predictive power (correlation > 0.1)")
    elif mean_corr > 0:
        print("  △ Weak positive correlation - may need improvement")
    else:
        print("  ✗ No predictive power - check target formulation and features")
    
    if overall_wfe > 0.8:
        print("  ✓ Good walk-forward efficiency (minimal overfitting)")
    else:
        print("  ⚠ Low WFE suggests overfitting - consider more regularization")
    
    print("=" * 95)


if __name__ == "__main__":
    run_training_pipeline()
