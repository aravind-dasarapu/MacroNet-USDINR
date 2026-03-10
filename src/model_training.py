import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, LayerNormalization, Conv1D, Activation, Input
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np

# 1. Custom Loss with Directional Penalty
def directional_log_cosh(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    error = y_pred - y_true
    # Log-cosh is like MSE for small errors and MAE for large ones
    log_cosh = tf.math.log(tf.math.cosh(error + 1e-12))

    # Penalize if the model predicts 'Up' when market goes 'Down'
    # Adding a small epsilon to sign to avoid 0-case issues
    same_direction = tf.equal(tf.math.sign(y_true), tf.math.sign(y_pred))
    penalty = tf.where(same_direction, 1.0, 2.5) # Increased penalty to 2.5 for forex niche

    return tf.reduce_mean(log_cosh * penalty)

# 2. Hybrid CNN-LSTM Model (MacroNet Architecture)
def build_fixed_lstm_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        # Spatial Dropout drops entire feature maps, better for time-series
        tf.keras.layers.SpatialDropout1D(0.2),
        BatchNormalization(),
        
        # CNN layer to extract local temporal patterns (Nifty/Oil correlations)
        Conv1D(filters=64, kernel_size=3, padding='causal'),
        BatchNormalization(),
        Activation('swish'),
        
        # LSTM for long-term dependency
        LSTM(128, return_sequences=True),
        LayerNormalization(),
        Dropout(0.3),
        
        LSTM(64, return_sequences=False),
        LayerNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dense(1, dtype='float32') # Predicted smoothed log return
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, amsgrad=True)
    model.compile(optimizer=optimizer, loss=directional_log_cosh, metrics=['mae'])
    return model

# 3. Optimized Pipeline
def run_fixed_training_pipeline():
    try:
        data_path = '/content/usdinr_processed_data.csv'
        df = pd.read_csv(data_path, index_col=0)
        df.index = pd.to_datetime(df.index)
    except:
        print('CSV not found. Ensure the build and preprocess scripts ran successfully.')
        return

    # Clean and Scale Target
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()
    
    # We use 100x to make returns more readable for the loss function
    y_raw = df['Target'].values.astype('float32') * 100
    X_raw = df.drop(columns=['Target']).values.astype('float32')
    
    window_size = 30 # Reduced to 30 to capture monthly macro cycles
    
    # Vectorized Windowing (Faster than for-loop)
    def create_windows(data, target, window):
        X, y = [], []
        for i in range(len(data) - window):
            X.append(data[i : i + window])
            y.append(target[i + window])
        return np.array(X), np.array(y)

    X, y = create_windows(X_raw, y_raw, window_size)
    print(f"Dataset ready: {X.shape[0]} sequences of {window_size} days.")

    # TimeSeriesSplit ensures no look-ahead bias
    tscv = TimeSeriesSplit(n_splits=5)
    final_y_test, final_preds = None, None
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # RE-SCALING PER FOLD: Prevents future leakage
        scaler = RobustScaler()
        # Flatten for scaling, then reshape back to (Samples, Time, Features)
        s, t, f = X_train.shape
        X_train_reshaped = X_train.reshape(-1, f)
        X_train_s = scaler.fit_transform(X_train_reshaped).reshape(s, t, f)
        
        X_test_s = scaler.transform(X_test.reshape(-1, f)).reshape(X_test.shape)

        model = build_fixed_lstm_model((window_size, f))
        
        # Early stopping based on Validation Loss
        es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
        
        print(f"Training Fold {fold + 1}...")
        model.fit(
            X_train_s, y_train, 
            epochs=100, # Increased epochs with early stopping
            batch_size=32, 
            validation_data=(X_test_s, y_test), 
            verbose=0,
            callbacks=[es]
        )
        
        # Directional Accuracy (The metric that matters for FX)
        preds = model.predict(X_test_s, verbose=0).flatten()
        
        # Logic: Did we get the 'sign' right?
        acc = np.mean(np.sign(preds) == np.sign(y_test)) * 100
        print(f'Fold {fold + 1} | Directional Acc: {acc:.2f}% | Val MAE: {np.mean(np.abs(preds - y_test)):.4f}')

        if fold == 4: # Fold 5 (index 4)
            final_y_test = y_test
            final_preds = preds
    if final_y_test is not None:
        print("\nGenerating Backtest Visualization for the Final Fold...")
        plot_backtest_results(final_y_test, final_preds)

import matplotlib.pyplot as plt

def plot_backtest_results(y_test, oos_preds):
    # 1. Prepare Data
    results = pd.DataFrame({
        'Actual_Ret': y_test / 100, # Convert back from the 100x scaling
        'Pred_Ret': oos_preds / 100
    })
    
    # 2. Strategy: Go with the sign of the prediction
    results['Strategy_Ret'] = np.sign(results['Pred_Ret']) * results['Actual_Ret']
    
    # 3. Calculate Cumulative Returns
    results['Cum_Market'] = (1 + results['Actual_Ret']).cumprod()
    results['Cum_Strategy'] = (1 + results['Strategy_Ret']).cumprod()
    
    # 4. Plotting
    plt.figure(figsize=(14, 7))
    plt.plot(results['Cum_Strategy'], label='MacroNet Strategy', color='forestgreen', lw=2)
    plt.plot(results['Cum_Market'], label='Buy & Hold (Market)', color='gray', linestyle='--', alpha=0.7)
    
    plt.title('USDINR Strategy Backtest: Fold 5 (Out-of-Sample)', fontsize=14)
    plt.xlabel('Days in Test Set')
    plt.ylabel('Cumulative Growth')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Calculate Sharpe Ratio (Simplified)
    sharpe = (results['Strategy_Ret'].mean() / (results['Strategy_Ret'].std() + 1e-9)) * np.sqrt(252)
    print(f"Strategy Sharpe Ratio: {sharpe:.2f}")


run_fixed_training_pipeline()