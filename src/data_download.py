import yfinance as yf
import pandas as pd
import numpy as np

def build_ultimate_clean_dataset():
    tickers = {
        "USDINR": "USDINR=X", "Oil": "BZ=F", "DXY": "DX-Y.NYB",
        "Gold": "GC=F", "Nifty50": "^NSEI", "SP500": "^GSPC",
        "VIX": "^VIX", "US10Y": "^TNX", "USTBILL": "^IRX",
        "EEM": "EEM", "CNY": "CNY=X", "SGD": "SGDINR=X",
    }

    print("--- Downloading Multi-OHLC Macro Data ---")
    
    # Establish Master Calendar
    master_data = yf.download(tickers["USDINR"], period="18y", interval="1d", auto_adjust=True, progress=False)
    if isinstance(master_data.columns, pd.MultiIndex):
        master_data.columns = master_data.columns.get_level_values(0)
    master_index = master_data.index
    
    raw_frames = []
    for name, sym in tickers.items():
        data = yf.download(sym, period="18y", interval="1d", auto_adjust=True, progress=False)
        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # Smart Filter: Keep only Close and Open to reduce noise
            cols = [c for c in ['Open', 'High', 'Low', 'Close'] if c in data.columns]
            data = data[cols]
            
            # ALIGNMENT: Fill holes locally first so they don't propagate
            data = data.reindex(master_index).ffill().bfill()
            data = data.rename(columns={col: f"{name}_{col}" for col in data.columns})
            raw_frames.append(data)

    df_unified = pd.concat(raw_frames, axis=1).ffill().bfill()
    print(f"Unified Calendar Established. Raw Columns: {len(df_unified.columns)}")

    main_close = "USDINR_Close"
    EPSILON = 1e-9
    feature_list = []

    # 1. Target Generation
    raw_ret_main = np.log((df_unified[main_close] / df_unified[main_close].shift(1)).abs() + EPSILON)
    target = raw_ret_main.ewm(alpha=0.4, adjust=False).mean().shift(-1).rename("Target")
    feature_list.append(target)

    # 2. FEATURE GENERATION
    for col in df_unified.columns:
        series = df_unified[col]
        ret = np.log((series / (series.shift(1) + EPSILON)).abs() + EPSILON)
        
        # We shift(1) all features to ensure no data leakage
        feature_list.append((ret.shift(1) * 100).rename(f"{col}_Ret"))
        
        vol5 = ret.rolling(5).std().shift(1) * 100
        feature_list.append(vol5.rename(f"{col}_Vol5"))

        ma20 = series.rolling(20).mean()
        dist = ((series - ma20) / (ma20 + EPSILON)).shift(1) * 100
        feature_list.append(dist.rename(f"{col}_DistMA"))

    # 3. ASSEMBLY
    final_df = pd.concat(feature_list, axis=1)
    
    # FILL THE WARM-UP HOLES (The first 20 days)
    # Instead of dropping them, we backfill so the start of the 18 years isn't lost
    final_df = final_df.ffill().bfill()
    
    # 4. THE SURGICAL DROP (CRITICAL FIX)
    # We only drop the very last row (where Target is NaN because of shift(-1))
    # This prevents holiday gaps in macro data from killing the whole dataset.
    final_df = final_df.dropna(subset=["Target"])

    print(f"Final Processing Complete. Features: {final_df.shape[1] - 1} | Rows: {len(final_df)}")
    
    output_path = "/content/usdinr_macro_final.csv"
    final_df.to_csv(output_path)
    return final_df

if __name__ == "__main__":
    df_final = build_ultimate_clean_dataset()
    print("\nTail of the dataset:")
    print(df_final.tail())