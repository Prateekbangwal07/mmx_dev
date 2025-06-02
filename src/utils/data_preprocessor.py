# utils/data_preprocessor.py

"""
Preprocessing class for Market Mix Modeling (MMX) use cases.
Includes adstock, saturation, lags, and normalization.
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("main_logger")


class DataPreprocessor:
    def __init__(self, adstock_alpha=0.5, saturation_beta=0.5):
        self.scaler = StandardScaler()
        self.adstock_alpha = adstock_alpha
        self.saturation_beta = saturation_beta
        self.fitted = False
        self.scale_cols = []

    def apply_adstock(self, series: pd.Series, alpha: float) -> pd.Series:
        """Apply adstock transformation with decay alpha."""
        logger.debug(f"Applying adstock with alpha={alpha}")
        result = []
        carryover = 0
        for val in series:
            carryover = val + alpha * carryover
            result.append(carryover)
        return pd.Series(result, index=series.index)

    def apply_saturation(self, series: pd.Series, beta: float) -> pd.Series:
        """Apply saturation (diminishing returns) using a power function."""
        logger.debug(f"Applying saturation with beta={beta}")
        return series**beta

    def create_lag_features(
        self, df: pd.DataFrame, columns: list, lags: list
    ) -> pd.DataFrame:
        """Create lagged features for selected columns."""
        for col in columns:
            for lag in lags:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)
        return df

    def fit(self, df: pd.DataFrame, scale_cols: list = None):
        """Fit scaler for selected columns."""
        self.scale_cols = (
            scale_cols
            if scale_cols
            else df.select_dtypes(include=[np.number]).columns.tolist()
        )
        self.scaler.fit(df[self.scale_cols].fillna(0))
        self.fitted = True
        logger.info("Scaler fitted on selected columns.")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply scaling to selected columns."""
        if not self.fitted:
            raise RuntimeError("Preprocessor not fitted. Call `.fit()` first.")

        df_scaled = df.copy()
        df_scaled[self.scale_cols] = self.scaler.transform(
            df_scaled[self.scale_cols].fillna(0)
        )
        return df_scaled

    def apply_mmx_transformations(
        self, df: pd.DataFrame, media_cols: list, lags: list = [1, 2, 3]
    ) -> pd.DataFrame:
        """
        Apply full MMX transformation pipeline:
        1. Adstock
        2. Saturation
        3. Lag features
        """
        df_transformed = df.copy()

        # Step 1: Adstock
        for col in media_cols:
            df_transformed[f"{col}_adstock"] = self.apply_adstock(
                df[col].fillna(0), self.adstock_alpha
            )

        # Step 2: Saturation
        for col in media_cols:
            adstock_col = f"{col}_adstock"
            df_transformed[f"{col}_saturated"] = self.apply_saturation(
                df_transformed[adstock_col], self.saturation_beta
            )

        # Step 3: Lag features
        df_transformed = self.create_lag_features(df_transformed, media_cols, lags)

        return df_transformed

    def fit_transform(self, df: pd.DataFrame, scale_cols: list = None) -> pd.DataFrame:
        self.fit(df, scale_cols)
        return self.transform(df)
