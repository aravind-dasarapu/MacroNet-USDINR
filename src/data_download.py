"""
Multi-asset financial data acquisition module for FX prediction.

This module downloads and aligns historical price data from multiple
financial instruments to build a unified dataset for USD/INR prediction.
"""
import os
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List
from pathlib import Path

from config_loader import get_config
from logger import setup_logger


class DataDownloader:
    """Handles multi-asset financial data download and initial processing."""
    
    def __init__(self, config_path: str = None):
        self.config = get_config(config_path)
        self.logger = setup_logger("data_download")
        
        self.data_dir = Path(self.config.data['data_dir'])
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.tickers = self.config.data['tickers']
        self.period = self.config.data['download_period']
        self.interval = self.config.data['download_interval']
        self.epsilon = self.config.preprocessing['epsilon']
    
    def download_ticker(self, symbol: str, name: str) -> pd.DataFrame:
        """
        Download data for a single ticker.
        
        Args:
            symbol: Yahoo Finance ticker symbol
            name: Display name for the asset
        
        Returns:
            DataFrame with OHLC data
        """
        try:
            data = yf.download(
                symbol,
                period=self.period,
                interval=self.interval,
                auto_adjust=True,
                progress=False
            )
            
            if data.empty:
                self.logger.warning(f"No data retrieved for {name} ({symbol})")
                return pd.DataFrame()
            
            # Handle MultiIndex columns from yfinance
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            self.logger.info(f"Downloaded {len(data)} rows for {name}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error downloading {name}: {str(e)}")
            return pd.DataFrame()
    
    def align_data(
        self, 
        raw_frames: List[pd.DataFrame], 
        master_index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Align all dataframes to a master calendar and handle missing values.
        
        Args:
            raw_frames: List of dataframes with different date ranges
            master_index: Reference date index for alignment
        
        Returns:
            Unified dataframe with aligned data
        """
        aligned_frames = []
        
        for df in raw_frames:
            if df.empty:
                continue
            
            # Reindex to master calendar
            df_aligned = df.reindex(master_index)
            
            # Forward fill then backward fill to handle gaps
            df_aligned = df_aligned.ffill().bfill()
            
            aligned_frames.append(df_aligned)
        
        if not aligned_frames:
            raise ValueError("No valid data frames to align")
        
        unified_df = pd.concat(aligned_frames, axis=1)
        
        # Final gap filling
        unified_df = unified_df.ffill().bfill()
        
        return unified_df
    
    def generate_features(self, df: pd.DataFrame, main_close: str) -> pd.DataFrame:
        """
        Generate technical features from raw price data.
        
        Args:
            df: Unified dataframe with all asset prices
            main_close: Column name for the target asset close price
        
        Returns:
            DataFrame with engineered features
        """
        feature_list = []
        
        # Generate target variable
        raw_return = np.log(
            (df[main_close] / df[main_close].shift(1)).abs() + self.epsilon
        )
        target = raw_return.ewm(
            alpha=self.config.preprocessing['ewm_alpha'], 
            adjust=False
        ).mean().shift(-1).rename("Target")
        
        feature_list.append(target)
        
        # Feature engineering for each column
        for col in df.columns:
            series = df[col]
            
            # Log returns
            returns = np.log(
                (series / (series.shift(1) + self.epsilon)).abs() + self.epsilon
            )
            
            # Shift by 1 to prevent data leakage
            feature_list.append(
                (returns.shift(1) * 100).rename(f"{col}_Ret")
            )
            
            # Rolling volatility
            vol_window = self.config.preprocessing['volatility_window']
            volatility = returns.rolling(vol_window).std().shift(1) * 100
            feature_list.append(volatility.rename(f"{col}_Vol{vol_window}"))
            
            # Distance from moving average
            ma_window = self.config.preprocessing['rolling_window']
            ma = series.rolling(ma_window).mean()
            distance = ((series - ma) / (ma + self.epsilon)).shift(1) * 100
            feature_list.append(distance.rename(f"{col}_DistMA"))
        
        final_df = pd.concat(feature_list, axis=1)
        
        # Handle warm-up period
        final_df = final_df.ffill().bfill()
        
        # Remove rows where target is missing
        final_df = final_df.dropna(subset=["Target"])
        
        return final_df
    
    def build_dataset(self) -> pd.DataFrame:
        """
        Main pipeline to download and process multi-asset data.
        
        Returns:
            Final processed dataframe ready for model training
        """
        self.logger.info("Starting multi-asset data download")
        
        # Establish master calendar using USDINR
        master_data = self.download_ticker(
            self.tickers["USDINR"], 
            "USDINR"
        )
        
        if master_data.empty:
            raise ValueError("Failed to download master calendar data (USDINR)")
        
        master_index = master_data.index
        self.logger.info(f"Master calendar established: {len(master_index)} trading days")
        
        # Download all tickers
        raw_frames = []
        for name, symbol in self.tickers.items():
            data = self.download_ticker(symbol, name)
            
            if not data.empty:
                # Keep only essential columns
                essential_cols = [c for c in ['Open', 'High', 'Low', 'Close'] 
                                  if c in data.columns]
                data = data[essential_cols]
                
                # Rename columns with asset prefix
                data = data.rename(
                    columns={col: f"{name}_{col}" for col in data.columns}
                )
                
                raw_frames.append(data)
        
        # Align all data to master calendar
        self.logger.info("Aligning data to master calendar")
        unified_df = self.align_data(raw_frames, master_index)
        self.logger.info(f"Unified dataset created: {unified_df.shape}")
        
        # Generate features
        self.logger.info("Generating technical features")
        main_close = "USDINR_Close"
        final_df = self.generate_features(unified_df, main_close)
        
        self.logger.info(
            f"Dataset complete: {final_df.shape[1] - 1} features, "
            f"{len(final_df)} samples"
        )
        
        # Save to disk
        output_path = self.data_dir / self.config.data['raw_file']
        final_df.to_csv(output_path)
        self.logger.info(f"Data saved to {output_path}")
        
        return final_df


def main():
    """Entry point for data download module."""
    downloader = DataDownloader()
    df = downloader.build_dataset()
    print(f"\nDataset shape: {df.shape}")
    print(f"\nFirst few rows:\n{df.head()}")
    print(f"\nLast few rows:\n{df.tail()}")


if __name__ == "__main__":
    main()