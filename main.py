#!/usr/bin/env python3
# main.py

"""
Main entry point for the application.

Author: Prateek Bangwal
Date: 2025-06-02
Description: A Python script for MMX and file logging.
"""

import argparse
import logging
import sys
import os
from datetime import datetime
from src.utils.data_utils import read_data
from src.utils.data_preprocessor import DataPreprocessor

import warnings

warnings.filterwarnings("ignore")

# Create logs directory if it doesn't exist
LOG_DIR = "logs"
SAVE_DIR = "data/output/"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

# Log file with timestamp
log_filename = os.path.join(
    LOG_DIR, f"script_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_filename, mode="w"),
    ],
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Main script entry point.")
    parser.add_argument(
        "--file", type=str, help="Path to configuration file", required=True
    )
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    logger.info("Starting the script...")

    # read the data
    try:
        df = read_data(args.file)
        logger.info(f"Data shape: {df.shape}")
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        return
    logger.info("Data loaded successfully.")
    media_cols = ["rtop_instapost_o.impressions", "rtop_twitterpost_o.impressions"]
    preprocessor = DataPreprocessor(adstock_alpha=0.6, saturation_beta=0.5)
    df_transformed = preprocessor.apply_mmx_transformations(df, media_cols, lags=[1, 2])
    df_transformed = preprocessor.fit_transform(df_transformed, scale_cols=media_cols)
    logger.info("Data preprocessing completed.")
    # Save the transformed data
    output_file = os.path.join(SAVE_DIR, "transformed_data.csv")
    df_transformed.to_csv(output_file, index=False)
    logger.info("Script finished successfully.")


if __name__ == "__main__":
    main()
