# MacroNet-USDINR: Multi-Asset Deep Learning FX Pipeline

**MacroNet-USDINR** is an institutional-grade machine learning framework designed to forecast USD/INR price direction. Unlike simple price-action models, this pipeline integrates global macro-economic drivers—including **Brent Crude Oil**, **US Dollar Index (DXY)**, and **Nifty 50**—to capture inter-market correlations.

---

## 🛠 Project Structure

```text
fx_prediction/
├── .venv/                # Virtual environment (Local only)
├── data/                 # Raw and Processed datasets
├── src/
│   ├── data_download.py  # Multi-asset ingestion via yfinance API
│   ├── preprocessing.py  # Feature engineering & temporal shifting
│   ├── model_training.py # Deep Neural Network implementation
│   └── main.py           # Pipeline orchestration
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation

🧠 Features & Technical Highlights
   Data Collection: Automated download of 15+ years of daily data for USD/INR, Brent Crude Oil, and the US Dollar Index.
   Stationary Feature Engineering: Implements Log-Returns and Normalized Indicators (Relative ATR, MACD/Price Ratio) to ensure data stationarity for superior model convergence.
   Temporal Integrity (Anti-Leakage): Rigorous application of $T-1$ feature lagging and $T+1$ target lead-shifting to eliminate Look-ahead Bias.
   Macro Correlation Engine: Analyzes rolling correlations between USD/INR and critical drivers like Oil (BZ=F) and DXY to detect global risk shifts.
   Signal Denoising: Utilizes Alpha-Smoothing (EWM) on target variables to filter out high-frequency market noise.
## Installation

1. Clone or download the project
2. Install dependencies:
   ```bash
      git clone [https://github.com/YourUsername/MacroNet-USDINR.git](https://github.com/YourUsername/MacroNet-USDINR.git)
cd fx_prediction
   pip install -r requirements.txt
   ```

## Usage

Run the complete pipeline:
```bash
# Create virtual environment
python -m venv .venv

# Activate environment (Windows)
.venv\Scripts\activate

# Activate environment (Linux/Mac)
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

cd src
python main.py
```

Or run individual steps:
```bash
python data_download.py
python data_preprocessing.py
python model_training.py
```

## Model Performance

- Directional Accuracy: ~56.3(Fold 3)%
- Validation MAE: 0.1136
- Target: Binary classification for price movements above a 0.0002 threshold
- Optimization: Uses Early Stopping and Learning Rate reduction on plateaus to prevent overfitting.

## Future Improvements

- Handle class imbalance
- Add more technical indicators
- Implement hyperparameter tuning
- Add prediction interface

🛡️ Data Flow & Integrity Check

   Download: Fetch financial data → data/market_data.csv

   Preprocess: Calculate indicators + Apply Shifts → data/processed_data.csv

   This project ensures that features are derived from historical data only. Every feature in the final dataset is shifted by at least one period relative to the target, ensuring that the model never sees the price of the day it is attempting to predict. This prevents the common "99% accuracy" trap caused by data leakage.

📈 Future Improvements

[ ] Implement LSTM/GRU recurrent layers for sequential memory.

[ ] Add Volatility GARCH modeling as an additional feature input.

[ ] Implement hyperparameter tuning (Keras Tuner).

[ ] Real-time inference dashboard using Streamlit.