"""
Utility functions for reading data files.
Supports CSV, Excel, and Pickle formats.
"""

import os
import pandas as pd
import logging

logger = logging.getLogger("main_logger")


def read_data(file_path: str) -> pd.DataFrame:
    """Reads data from CSV, Excel, Pickle or parquet file based on extension."""
    logger.info(f"Reading file: {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".csv":
            return pd.read_csv(file_path)
        elif ext in [".xls", ".xlsx"]:
            return pd.read_excel(file_path)
        elif ext == ".pkl":
            return pd.read_pickle(file_path)
        elif ext == ".parquet":
            return pd.read_parquet(file_path)
        else:
            logger.error(f"Unsupported file format: {ext}")
            raise ValueError(f"Unsupported file format: {ext}")
    except Exception as e:
        logger.exception(f"Failed to read file: {e}")
        raise


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Saves DataFrame to CSV, Excel, Pickle or parquet file based on extension."""
    logger.info(f"Saving DataFrame to file: {file_path}")
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".csv":
            df.to_csv(file_path, index=False)
        elif ext in [".xls", ".xlsx"]:
            df.to_excel(file_path, index=False)
        elif ext == ".pkl":
            df.to_pickle(file_path)
        elif ext == ".parquet":
            df.to_parquet(file_path, index=False)
        else:
            logger.error(f"Unsupported file format for saving: {ext}")
            raise ValueError(f"Unsupported file format for saving: {ext}")
    except Exception as e:
        logger.exception(f"Failed to save file: {e}")
        raise
