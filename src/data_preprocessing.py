"""
Data preprocessing and feature selection for FX prediction model.

This module performs correlation-based feature selection and adds
technical overlays to prepare data for model training.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from config_loader import get_config
from logger import setup_logger


class DataPreprocessor:
    """Handles feature selection and final data preparation."""

    def __init__(self, config_path: str = None):
        self.config = get_config(config_path)
        self.logger = setup_logger("data_preprocessing")

        self.data_dir = Path(self.config.data["data_dir"])
        self.corr_threshold = self.config.preprocessing["correlation_threshold"]
        self.min_features = self.config.preprocessing["min_features"]
        self.top_features = self.config.preprocessing["top_features"]
        self.epsilon = self.config.preprocessing["epsilon"]

    def load_raw_data(self) -> pd.DataFrame:
        """Load raw data from CSV file."""
        input_path = self.data_dir / self.config.data["raw_file"]

        if not input_path.exists():
            raise FileNotFoundError(
                f"Raw data file not found: {input_path}. Run data_download.py first."
            )

        df = pd.read_csv(input_path, index_col=0)
        df.index = pd.to_datetime(df.index)

        self.logger.info(f"Loaded data: {df.shape}")
        return df

    def find_main_price_column(self, df: pd.DataFrame) -> str:
        """
        Identify the main price column for USDINR.

        Args:
            df: Input dataframe

        Returns:
            Column name for USDINR close price
        """
        candidates = ["USDINR_Close", "Close_USDINR=X"]

        for candidate in candidates:
            if candidate in df.columns:
                return candidate

        # Fallback: search for any column with both USDINR and Close
        matching_cols = [c for c in df.columns if "USDINR" in c and "Close" in c]

        if not matching_cols:
            raise ValueError("Could not find USDINR close price column")

        return matching_cols[0]

    def select_features_by_correlation(
        self, df: pd.DataFrame, target: pd.Series
    ) -> Tuple[pd.DataFrame, list]:
        """
        Select features based on correlation with target.

        Uses return-based correlation to identify predictive features.

        Args:
            df: Feature dataframe
            target: Target series

        Returns:
            Tuple of (selected features dataframe, feature names list)
        """
        # Calculate returns for correlation analysis
        df_returns = df.drop(columns=["Target"]).pct_change().fillna(0)

        # Compute correlations with target
        correlations = df_returns.corrwith(target).abs().sort_values(ascending=False)

        self.logger.info(f"Top 10 correlations:\n{correlations.head(10)}")

        # Select features meeting threshold
        relevant_features = correlations[
            correlations > self.corr_threshold
        ].index.tolist()

        # Ensure minimum number of features
        if len(relevant_features) < self.min_features:
            self.logger.warning(
                f"Only {len(relevant_features)} features met threshold. "
                f"Selecting top {self.top_features} instead."
            )
            relevant_features = correlations.head(self.top_features).index.tolist()

        self.logger.info(f"Selected {len(relevant_features)} features")

        selected_df = df[relevant_features].copy()

        return selected_df, relevant_features

    def add_technical_overlays(
        self, df: pd.DataFrame, full_df: pd.DataFrame, main_close_col: str
    ) -> pd.DataFrame:
        close = full_df[main_close_col]
        high = full_df[main_close_col.replace("Close", "High")]
        low = full_df[main_close_col.replace("Close", "Low")]

        # --- 1. ATR (Volatility) ---
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["ATR"] = tr.rolling(window=14).mean().shift(1)
        df["Norm_ATR"] = (df["ATR"] / close.shift(1)) * 100

        # --- 2. RSI (Momentum) ---
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + self.epsilon)
        df["RSI"] = (100 - (100 / (1 + rs))).shift(1)

        # --- 3. MACD (Trend) ---
        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        df["MACD_Hist"] = (macd - signal).shift(1)

        # --- 4. ADX (Trend Strength - The Regime Fix) ---
        plus_dm = high.diff().where(
            (high.diff() > low.diff().abs()) & (high.diff() > 0), 0
        )
        minus_dm = (
            low.diff()
            .abs()
            .where((low.diff().abs() > high.diff()) & (low.diff().abs() > 0), 0)
        )

        smooth_plus_dm = plus_dm.rolling(window=14).mean()
        smooth_minus_dm = minus_dm.rolling(window=14).mean()

        di_plus = 100 * (smooth_plus_dm / (df["ATR"] + self.epsilon))
        di_minus = 100 * (smooth_minus_dm / (df["ATR"] + self.epsilon))

        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus + self.epsilon)
        df["ADX"] = dx.rolling(window=14).mean().shift(1)

        return df

    def preprocess_data(self) -> pd.DataFrame:
        """
        Main preprocessing pipeline.

        Returns:
            Final preprocessed dataframe ready for model training
        """
        self.logger.info("Starting data preprocessing")

        # Load data
        df = self.load_raw_data()

        # Find main price column
        main_close_col = self.find_main_price_column(df)
        self.logger.info(f"Using {main_close_col} as main price column")

        # Extract target
        target = df["Target"].copy()

        # Feature selection
        selected_df, feature_names = self.select_features_by_correlation(df, target)

        # Add target back
        selected_df["Target"] = target

        # Add technical overlays
        selected_df = self.add_technical_overlays(selected_df, df, main_close_col)

        # Handle missing values
        final_target = selected_df.pop("Target")
        selected_df = selected_df.ffill().bfill()
        selected_df["Target"] = final_target

        # Remove rows with missing target
        final_df = selected_df.dropna(subset=["Target"])

        self.logger.info(
            f"Preprocessing complete: {final_df.shape[1] - 1} features, "
            f"{len(final_df)} samples"
        )

        # Save processed data
        output_path = self.data_dir / self.config.data["processed_file"]
        final_df.to_csv(output_path)
        self.logger.info(f"Processed data saved to {output_path}")

        return final_df


def main():
    """Entry point for preprocessing module."""
    preprocessor = DataPreprocessor()
    df = preprocessor.preprocess_data()
    print(f"\nProcessed dataset shape: {df.shape}")
    print(f"\nFeatures: {list(df.columns)}")
    print(f"\nSample data:\n{df.head()}")


if __name__ == "__main__":
    main()
