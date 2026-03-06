import pandas as pd
import numpy as np
import os

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD and Signal Line"""
    ema_fast = close.ewm(span=fast).mean()
    ema_slow = close.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_width(close, period=20, num_std=2):
    """Calculate Bollinger Bands Width"""
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    width = (upper - lower) / sma
    return width

def calculate_stochastic(high, low, close, period=14):
    """Calculate Stochastic K%"""
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
    return k_percent

def calculate_roc(close, period):
    """Calculate Rate of Change"""
    return close.pct_change(periods=period) * 100

def calculate_rolling_correlation(series1, series2, period=14):
    """Calculate Rolling Correlation"""
    return series1.rolling(window=period).corr(series2)

def apply_target_smoothing(df, window=3, alpha=0.3):
    """
    Applies smoothing to the Target variable to reduce noise.
    """
    # Option A: Simple Centered Moving Average (Balanced)
    # This looks 1 day back and 1 day forward to find the true "path"
    df['Target_SMA'] = df['Target'].rolling(window=window, center=True).mean()

    # Option B: Exponential Smoothing (Weighted towards recent)
    # Alpha 0.3 means 30% weight to current day, 70% to previous trend
    df['Target_EMA'] = df['Target'].ewm(alpha=alpha, adjust=False).mean()
    
    # We use the smoothed version for training, but keep original for evaluation
    # To use it, simply swap the target column name
    df['Target'] = df['Target_EMA'] 
    
    return df.dropna()
def preprocess_data():
    input_path = os.path.join(os.path.dirname(__file__), "../data/data.csv")
    df = pd.read_csv(input_path, index_col=0)

    # Extract OHLC for each ticker
    close = df['Close_USDINR=X']
    high = df['High_USDINR=X']
    low = df['Low_USDINR=X']
    oil = df['Close_BZ=F']
    dxy = df['Close_DX-Y.NYB']

    # 1. Log Returns (Stationarity)
    df['USDINR_Ret'] = np.log(close / close.shift(1)).shift(1)

    # 2. RSI (Momentum Indicator)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss))).shift(1)


    # 3. Moving Average crossover (Trend Indicator)
    # Relative Volatility (ATR / Price)
    df['ATR_Rel'] = (calculate_atr(high, low, close) / close).shift(1)
    # MACD Histogram Relative to Price
    macd_series, signal_series, _ = calculate_macd(close)
    df['MACD_Rel'] = (macd_series / close).shift(1)

    # Bollinger Width (Already a ratio)
    df['BB_Width'] = calculate_bollinger_width(close).shift(1)  


    # Correlations (Stationary -1 to 1)
    df['Corr_Oil'] = close.rolling(14).corr(oil).shift(1)  # Shifted to prevent lookahead bias
    df['Corr_DXY'] = close.rolling(14).corr(dxy).shift(1)

    threshold = 0.0002  
    # 15. Target (Predict the next return)
    df['Target'] = (df['USDINR_Ret']> threshold).astype(int).shift(-1)

    df['Vol_5'] = df['Target'].rolling(window=5).std().shift(1)
    
    # --- NEW: Volatility Momentum ---
    # This tells the model if the market is "waking up"
    df['Vol_15'] = df['Target'].rolling(window=15).std().shift(1)
    df['Vol_Trend'] = df['Vol_5'] - df['Vol_15']
    

    # df['RSI'] = df['RSI'].shift(1)
    # df['ATR_Rel'] = df['ATR_Rel'].shift(1)
    # df['MACD'] # Neutral RSI for missing values
    # # --- Final Cleanup ---

    df = df.ffill().bfill().dropna()
    df = apply_target_smoothing(df)

    output_path = os.path.join(os.path.dirname(__file__), "../data/processed_data.csv")
    df.to_csv(output_path)
    print(f"✅ Enhanced Preprocessing: Created {len(df.columns)} features including ATR, MACD, Bollinger Bands, ROC, Stochastic, and correlations.")