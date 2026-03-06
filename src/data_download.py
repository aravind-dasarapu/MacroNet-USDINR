import yfinance as yf
import pandas as pd
import numpy as np
import os

def build_ultimate_clean_dataset(target_thresh=0.01, redundancy_thresh=0.90):
    tickers = {
        "USDINR": "USDINR=X", "Oil": "BZ=F", "DXY": "DX-Y.NYB",
        "Gold": "GC=F", "Nifty50": "^NSEI", "SP500": "^GSPC",
        "VIX": "^VIX", "US10Y": "^TNX", "USTBILL": "^IRX",
        "EEM": "EEM", "CNY": "CNY=X", "SGD": "SGDINR=X"
    }

    print("📥 Downloading data from Yahoo Finance...")
    # Download all at once - more efficient
    raw_data = yf.download(list(tickers.values()), period="15y", interval="1d", auto_adjust=True)

    # # --- FIX: Handle Multi-Index and Extract Close Prices ---
    if isinstance(raw_data.columns, pd.MultiIndex):
        raw_data.columns = [f"{col[0]}_{col[1]}" for col in raw_data.columns] 

    # # Fill gaps in global market data (holidays)
    df_raw = raw_data.ffill().bfill()

    main_ticker = "Close_USDINR=X"
    if main_ticker not in df_raw.columns:
        main_ticker = [c for c in df.raw_columns if 'USDINR=X' in c and "Close" in c][0]

    EPSILON = 1e-9
    # # --- 2. THE TARGET (The Future) ---
    ratio = (df_raw[main_ticker] / df_raw[main_ticker].shift(1)).abs()
    raw_returns = np.log(ratio + EPSILON)
    smoothed_ret = raw_returns.ewm(alpha=0.4, adjust=False).mean()
    # # Shift -1 means today's row contains TOMORROW'S smoothed return
    
    # # --- 3. FEATURE ENGINEERING (The Past) ---
    feature_list = []
    
    target_df = pd.DataFrame({'Target': smoothed_ret.shift(-1)}, index=df_raw.index)
    feature_list.append(target_df)

    feature_list.append(df_raw[[main_ticker]].rename(columns={main_ticker: 'Close_USDINR=X'}))

    for column_name in df_raw.columns:
        if "Close" in column_name:
            ratio_col = (df_raw[column_name] / df_raw[column_name].shift(1)).abs()
            ret = np.log(ratio_col + EPSILON)
        # Create the feature bundle for this ticker
            temp_df = pd.DataFrame(index=df_raw.index)
            temp_df[f'{column_name}_Ret'] = ret.shift(1)
            temp_df[f'{column_name}_Vol5'] = ret.rolling(5).std().shift(1)
            temp_df[f'{column_name}_Vol15'] = ret.rolling(15).std().shift(1)
            temp_df[f'{column_name}_VolTrend'] = temp_df[f'{column_name}_Vol5'] - temp_df[f'{column_name}_Vol15']
        
            feature_list.append(temp_df)    
    # # 4. RSI Calculation
    def calc_rsi(series, period=14):
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=period-1, adjust=False).mean()
        ema_down = down.ewm(com=period-1, adjust=False).mean()
        return 100 - (100 / (1 + (ema_up / ema_down)))

    feature_list.append(pd.DataFrame({'USDINR_RSI': calc_rsi(df_raw[main_ticker]).shift(1)}, index=df_raw.index))

    # # 5. FINAL CLEANUP
    df = pd.concat(feature_list, axis=1).dropna()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # # --- 6. SMART FILTERING ---
    # # Don't drop 'Actual_Close' and 'Target' during filtering
    
    # # Correlation with Target


    features_only = df.drop(columns=['Target', 'Close_USDINR=X'])
    correlations = features_only.corrwith(df['Target'])

    selected_features = correlations[correlations.abs() >= target_thresh].index.tolist()
    
    # # # Redundancy Check
    feature_corr = features_only[selected_features].corr().abs()
    upper = feature_corr.where(np.triu(np.ones(feature_corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > redundancy_thresh)]
    
    final_features = [f for f in selected_features if f not in to_drop]
    final_df = pd.concat([df[['Target','Close_USDINR=X']], features_only[final_features], df_raw[['High_USDINR=X', 'Low_USDINR=X', 'Close_BZ=F', 'Close_DX-Y.NYB']]],  axis=1).dropna()

    # # # --- SAVE ---
    output_path = os.path.join(os.path.dirname(__file__), "../data/data.csv")

    # # output_path = os.path.join(data_dir, "processed_data.csv")

    final_df.to_csv(output_path)
    
    print(f"✅ Saved to: {output_path}")
    print(f"First five rows:\n{final_df.head()}")
    print(f"📊 Final Shape: {final_df.shape} (Includes Actual_Close & Target)")
    return final_df
    # print(df)


if __name__ == "__main__":
    build_ultimate_clean_dataset()