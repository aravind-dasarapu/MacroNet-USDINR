"""
Main pipeline orchestrator for FX prediction system.

Coordinates data download, preprocessing, and model training phases
with proper error handling and logging.
"""
import os
import sys
from pathlib import Path
import argparse

from config_loader import get_config
from logger import setup_logger
from data_download import DataDownloader
from data_preprocessing import DataPreprocessor
from model_training import ModelTrainer


class Pipeline:
    """Main pipeline controller."""
    
    def __init__(self, config_path: str = None, use_wandb: bool = False):
        self.config = get_config(config_path)
        self.logger = setup_logger("pipeline")
        self.use_wandb = use_wandb
        
        self.data_dir = Path(self.config.data['data_dir'])
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def check_data_exists(self) -> bool:
        """Check if processed data already exists."""
        processed_path = self.data_dir / self.config.data['processed_file']
        return processed_path.exists()
    
    def run(self, skip_download: bool = False, skip_preprocessing: bool = False):
        """
        Execute the complete FX prediction pipeline.
        
        Args:
            skip_download: Skip data download if data exists
            skip_preprocessing: Skip preprocessing if processed data exists
        """
        self.logger.info("="*60)
        self.logger.info("FX Prediction Pipeline Starting")
        self.logger.info("="*60)
        
        try:
            # Step 1: Data Download
            if skip_download and self.check_data_exists():
                self.logger.info("Skipping data download (data exists)")
            else:
                self.logger.info("\n[Step 1/3] Downloading financial data")
                self.logger.info("-"*60)
                downloader = DataDownloader()
                downloader.build_dataset()
                self.logger.info("Data download completed successfully")
            
            # Step 2: Preprocessing
            processed_path = self.data_dir / self.config.data['processed_file']
            if skip_preprocessing and processed_path.exists():
                self.logger.info("\nSkipping preprocessing (processed data exists)")
            else:
                self.logger.info("\n[Step 2/3] Preprocessing data")
                self.logger.info("-"*60)
                preprocessor = DataPreprocessor()
                preprocessor.preprocess_data()
                self.logger.info("Preprocessing completed successfully")
            
            # Step 3: Model Training
            self.logger.info("\n[Step 3/3] Training model")
            self.logger.info("-"*60)
            trainer = ModelTrainer(use_wandb=self.use_wandb)
            summary = trainer.train_model()
            self.logger.info("Model training completed successfully")
            
            # Final summary
            self.logger.info("\n" + "="*60)
            self.logger.info("Pipeline Completed Successfully!")
            self.logger.info("="*60)
            self.logger.info(f"Experiment: {summary['experiment_name']}")
            self.logger.info(
                f"Average Directional Accuracy: "
                f"{summary['average_directional_accuracy']:.2f}%"
            )
            self.logger.info(f"Average MAE: {summary['average_mae']:.4f}")
            self.logger.info(
                f"Model saved to: {self.config.deployment['model_dir']}"
            )
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            raise


def main():
    """Command-line interface for the pipeline."""
    parser = argparse.ArgumentParser(
        description="FX Prediction Pipeline for USD/INR"
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip data download if data exists'
    )
    
    parser.add_argument(
        '--skip-preprocessing',
        action='store_true',
        help='Skip preprocessing if processed data exists'
    )
    
    parser.add_argument(
        '--use-wandb',
        action='store_true',
        help='Enable Weights & Biases experiment tracking'
    )
    
    parser.add_argument(
        '--download-only',
        action='store_true',
        help='Only download data, skip preprocessing and training'
    )
    
    parser.add_argument(
        '--preprocess-only',
        action='store_true',
        help='Only preprocess data, skip training'
    )
    
    args = parser.parse_args()
    
    # Handle individual steps
    if args.download_only:
        downloader = DataDownloader(args.config)
        downloader.build_dataset()
        return
    
    if args.preprocess_only:
        preprocessor = DataPreprocessor(args.config)
        preprocessor.preprocess_data()
        return
    
    # Run full pipeline
    pipeline = Pipeline(args.config, args.use_wandb)
    pipeline.run(
        skip_download=args.skip_download,
        skip_preprocessing=args.skip_preprocessing
    )


if __name__ == "__main__":
    main()