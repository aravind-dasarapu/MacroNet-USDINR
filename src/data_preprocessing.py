import pandas as pd
import numpy as np

def preprocess_data(corr_threshold=0.012):
    input_path = "/content/usdinr_macro_final.csv"
    try:
        # 1. LOAD DATA (We know this has 4666 rows)
        df = pd.read_csv(input_path, index_col=0)
        df.index = pd.to_datetime(df.index)
        print(f"Starting Preprocessing with {len(df)} rows.")
    except Exception as e:
        print(f"Error: {e}")
        return

    # 2. FIND THE MAIN PRICE COLUMN
    main_close_col = next((c for c in ["USDINR_Close", "Close_USDINR=X"] if c in df.columns), None)
    if not main_close_col:
        main_close_col = [c for c in df.columns if "USDINR" in c and "Close" in c][0]

    # 3. STATIONARY CORRELATION TEST (The Logic Fix)
    # We create a temporary returns-only dataframe to find 'the winners'
    # Comparing Return-to-Return is the only way to find forecasting signal
    target = df["Target"].copy()
    df_returns_test = df.drop(columns=["Target"]).pct_change().fillna(0)
    
    # Calculate correlation: Return of Feature vs. Return of USDINR
    correlations = df_returns_test.corrwith(target).abs().sort_values(ascending=False)
    
    # Select features that meet threshold
    relevant_features = correlations[correlations > corr_threshold].index.tolist()
    
    # SAFETY: Ensure we don't wipe the dataset
    if len(relevant_features) < 15:
        print("Correlation threshold too strict for returns. Picking Top 50 signals.")
        relevant_features = correlations.head(50).index.tolist()

    # 4. RECONSTRUCT DATASET
    # We take the raw prices/values of the selected relevant features
    processed = df[relevant_features].copy()
    processed["Target"] = target
    
    # 5. ADD TECHNICAL OVERLAYS (Memory & Momentum)
    close = df[main_close_col]
    log_ret = np.log(close / (close.shift(1) + 1e-9))
    
    # Add Lags and Distances
    processed["USDINR_Lag1"] = log_ret.shift(1) * 100
    # Use a 20-day SMA; 50 is often too long for daily forex returns
    sma20 = close.rolling(20).mean()
    processed["Dist_SMA20"] = ((close - sma20) / (sma20 + 1e-9)).shift(1) * 100

    # 6. MAXIMUM RETENTION CLEANUP (No Suicide Drop)
    # A. Move target out
    final_target = processed.pop("Target")
    
    # B. Bridge the 20-day SMA 'warm-up' holes and any holiday gaps
    processed = processed.ffill().bfill()
    
    # C. Re-attach Target
    processed["Target"] = final_target
    
    # D. SURGICAL DROP: Only drop where Target is missing (the very last row)
    final_df = processed.dropna(subset=["Target"])

    print(f"--- PREPROCESSING COMPLETE ---")
    print(f"Features Selected: {final_df.shape[1]-1}")
    print(f"Total Rows Retained: {len(final_df)}") # Should be ~4665
    
    final_df.to_csv("/content/usdinr_processed_data.csv")
    return final_df

if __name__ == "__main__":
    processed_df = preprocess_data(corr_threshold=0.01)