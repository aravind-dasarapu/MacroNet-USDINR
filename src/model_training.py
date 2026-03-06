import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, LeakyReLU, GaussianNoise, BatchNormalization, Conv1D, MaxPooling1D, Flatten, concatenate, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

# 1. HARDWARE SETTINGS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


# def volatility_weighted_loss(y_true, y_pred):
#     # Standard MSE
#     squared_difference = tf.square(y_true - y_pred)
    
#     # Increase weight when the true target move is large (High Volatility)
#     # This prevents the model from settling for a "Safe Average"
#     weight = tf.abs(y_true) + 0.1 # The 0.1 is a base weight
    
#     return tf.reduce_mean(squared_difference * weight)    


def build_lstm_model(input_shape):

    # inputs = Input(shape=input_shape),


    # cnn = Conv1D(filters  = 16, kernel_size=2, activation='relu')(inputs)
    # cnn = Flatten()(cnn)

    # LSTM(64, return_sequences=True, 
    #      kernel_regularizer=l2(1e-6), recurrent_regularizer=l2(1e-6),),
    # BatchNormalization(),
    # Dropout(0.2),
    
    # lstm = LSTM(32, return_sequences=False, 
    #         kernel_regularizer=l2(1e-6), recurrent_regularizer=l2(1e-6))(inputs)
    
    # merged = concatenate([cnn, lstm])

    # # LeakyReLU ensures the model doesn't "die" and output zero
    # x = Dense(16, activation='relu')(merged)
    # x = Dropout(0.1)(x)
    # outputs = Dense(1)(x) 
    
    # model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # # Higher learning rate to help the model escape the "flat line" local minima

    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=16, kernel_size=2, activation='relu', padding='causal'),
        BatchNormalization(),

        LSTM(64, return_sequences=False, kernel_regularizer=l2(1e-6), recurrent_regularizer=l2(1e-6)),
        BatchNormalization(),
        
        Dense(16, activation='relu'),
        LeakyReLU(alpha=0.1),
        Dropout(0.2),
        Dense(1, dtype='float32')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='Huber', metrics=['mae'])
    return model

def run_training_pipeline():
    try:
        data_path = os.path.join(os.path.dirname(__file__), "../data/data.csv")
        df = pd.read_csv(data_path, index_col=0)
    except:
        print("CSV not found.")
        return

    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna().astype('float32')

    # --- CHANGE 1: SCALE TARGET ---
    # Multiplying by 100 makes the returns visible to the model's loss function
    target = df['Target'].values * 100 
    features = df.drop(columns=['Target']).values 
    raw_target = df['Target'].values
    smoothed_target = df['Target'].ewm(alpha=0.3).mean().values

    window_size = 5
    X, y = [], []
    for i in range(len(features) - window_size):
        X.append(features[i : i + window_size])
        y.append(target[i + window_size])
    X, y = np.array(X), np.array(y)

    tscv = TimeSeriesSplit(n_splits=3)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        print(f"\n--- FOLD {fold + 1} ---")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        scaler = RobustScaler()
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
        X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

        X_train_scaled = np.nan_to_num(X_train_scaled)
        X_test_scaled = np.nan_to_num(X_test_scaled)

        model = build_lstm_model((X_train_scaled.shape[1], X_train_scaled.shape[2]))
        
        # --- CHANGE 2: CALLBACKS ---
        # ReduceLROnPlateau helps if the model gets stuck
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5)
        early_stop = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)

        model.fit(X_train_scaled, y_train, 
                  epochs=150, # 200 might be too much, 150 is a sweet spotc
                  batch_size=64, 
                  validation_data=(X_test_scaled, y_test),
                  callbacks=[reduce_lr, early_stop],
                  verbose=1)

        # --- CHANGE 3: REVERSE SCALING FOR PLOT ---
        y_pred_scaled = model.predict(X_test_scaled, verbose=0).flatten()
        
        # Scale back to original size for accurate MAE and plotting
        y_pred = y_pred_scaled / 100
        y_test_orig = y_test / 100
        
        y_test_actual_raw = raw_target[test_idx]
        correct_direction = np.sign(y_pred) == np.sign(y_test_actual_raw)
        accuracy = np.mean(correct_direction) * 100
        
        print(f"✅ Fold {fold + 1} Directional Accuracy: {accuracy:.2f}%")
        
        # 3. Visualization
        plt.figure(figsize=(15, 6))
        zoom_range = 100 
        plt.plot(y_test_orig[-zoom_range:], label='Actual Returns', color='black', alpha=0.5)
        plt.plot(y_pred[-zoom_range:], label='LSTM Prediction (Bold)', color='red', linewidth=1.5)
        plt.axhline(0, color='blue', linestyle='--', alpha=0.3)
        plt.title(f"Fold {fold+1} | Accuracy: {accuracy:.2f}% | MAE: {mean_absolute_error(y_test_orig, y_pred):.6f}")
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.show()

if __name__ == "__main__":
    run_training_pipeline()
