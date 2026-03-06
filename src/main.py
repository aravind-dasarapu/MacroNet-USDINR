import os
import pandas as pd
from data_download import build_ultimate_clean_dataset
from data_preprocessing import preprocess_data
from model_training import run_training_pipeline

def main():
    # Define the path where the final cleaned data should live
    # Adjust this path to match exactly where your 'build_ultimate_clean_dataset' saves the file
    DATA_DIR = "../data/processed/"
    FILE_NAME = "processed_data.csv"
    data_path = os.path.join(os.path.dirname(__file__), DATA_DIR, FILE_NAME)

    print("🚀 Starting the FX prediction pipeline...")

    # --- LOGIC: CHECK IF DATA EXISTS ---
    if os.path.exists(data_path):
        print(f"✅ Found existing data at {data_path}. Skipping download.")
        # Optional: Load a snippet to verify integrity
        df_check = pd.read_csv(data_path, nrows=5)
        print(f"📊 Data Preview (First 5 rows):\n{df_check.head()}")
    else:
        print("⚠️ Data not found locally.")
        
        # Step 1: Download data
        print("Step 1: Downloading financial data from Yahoo Finance...")
        # Ensure build_ultimate_clean_dataset() saves to data_path internally
        build_ultimate_clean_dataset() 

        # Step 2: Preprocess data
        # Note: If Step 1 already produced the 'processed_data.csv', 
        # check if your preprocess_data() is redundant or adds extra scaling/windowing.
        print("Step 2: Performing final feature engineering and cleaning...")
        preprocess_data()

    # Step 3: Train model
    # The training pipeline will now pick up the file regardless of if it was just made or already existed.
    print("Step 3: Training the LSTM model on Intel iGPU...")
    run_training_pipeline()

    print("✨ Pipeline completed successfully!")

if __name__ == "__main__":
    main()