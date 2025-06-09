import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import bambi as bmb
import numpy as np
import arviz as az


class ModelConfig:
    """
    Handles loading and parsing of the YAML configuration file for the model pipeline.
    """
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self):
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file {self.config_path} not found.")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")

    def get_model_config(self):
        return self.config


class ModelStage:
    """
    Represents a single stage in the model pipeline with support for various model types.
    """
    def __init__(self, stage_name, config, data):
        self.stage_name = stage_name
        self.data = data
        self.output_column = config["output_column"]
        self.model_type = config["model_type"]  # Added to select model type
        self.independent_vars = config.get("independent_vars", [])
        self.hyperparams = config.get("hyperparams", {})
        self.use_lagged_effects = config.get("use_lagged_effects", False)
        self.lagged_features = config.get("lagged_features", {})
        self.fit_params = config.get("fit_params", {})
        self.model = None
        self.fit_result = None

        if not self.independent_vars:
            raise ValueError(f"Stage {stage_name} requires independent variables.")

    def add_weighted_lagged_effects(self):
        """
        Adds weighted lagged features to the dataset for this stage.
        """
        for column, settings in self.lagged_features.items():
            lags = settings["lags"]
            weights = settings["weights"]

            if len(weights) != lags:
                raise ValueError(f"Number of weights must match the number of lags for {column}.")

            for lag in range(1, lags + 1):
                self.data[f"{column}_lag{lag}"] = self.data[column].shift(lag)

            lagged_cols = [f"{column}_lag{lag}" for lag in range(1, lags + 1)]
            self.data[f"{column}_weighted"] = sum(
                self.data[col] * weight for col, weight in zip(lagged_cols, weights)
            )

            self.data.fillna(0, inplace=True)

    def fit(self):
        """
        Fits the model for this stage using the specified fit parameters.
        """
        if self.use_lagged_effects:
            print(f"Adding lagged effects for {self.stage_name}.")
            self.add_weighted_lagged_effects()

        X = self.data[self.independent_vars]
        y = self.data[self.output_column]

        print(f"Fitting {self.model_type} model for {self.stage_name} with params {self.hyperparams}")

        # Fit model based on model_type
        try:
            if self.model_type == "bambi":
                self.model = bmb.Model(f"{self.output_column} ~ {' + '.join(self.independent_vars)}", data=self.data)
                self.fit_result = self.model.fit(**self.fit_params)
                predictions = self.model.predict(idata=self.fit_result, data=self.data, inplace=False)
                final_preds = az.summary(predictions).reset_index()["mean"].values[-len(self.data):]
                #self.data[self.output_column] = (self.fit_result.posterior["Intercept"].mean().values + self.fit_result.posterior["media_channel_clicks_weighted"].mean().values * self.data["media_channel_clicks_weighted"] + self.fit_result.posterior["other_channel_clicks"].mean().values * self.data["other_channel_clicks"])
                self.data[self.output_column] = final_preds

            elif self.model_type == "linear_regression":
                self.model = LinearRegression(**self.hyperparams)
                self.model.fit(X, y)
                self.data[self.output_column] = self.model.predict(X)

            elif self.model_type == "lasso":
                self.model = Lasso(**self.hyperparams)
                self.model.fit(X, y)
                self.data[self.output_column] = self.model.predict(X)

            elif self.model_type == "svm":
                self.model = SVR(**self.hyperparams)
                self.model.fit(X, y)
                self.data[self.output_column] = self.model.predict(X)

            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        except Exception as e:
            raise RuntimeError(f"Error fitting model for stage {self.stage_name}: {e}")

    def get_fit_result(self):
        return self.fit_result


class ModelPipeline:
    """
    A pipeline to dynamically fit and predict multiple models based on configuration.
    """
    def __init__(self, data, config_path):
        self.data = data
        self.config = ModelConfig(config_path)
        self.stages = self._initialize_stages()

    def _initialize_stages(self):
        """
        Creates ModelStage objects for each stage in the configuration.
        """
        model_stage_config = self.config.get_model_config().get("model_stage", {})
        stages = {}
        for stage_name, stage_config in model_stage_config.items():
            stages[stage_name] = ModelStage(stage_name, stage_config, self.data)
        return stages

    def run(self):
        for stage_name, stage in self.stages.items():
            print(f"Running stage: {stage_name}")
            stage.fit()
            self.data = stage.data  # Update pipeline data for subsequent stages
        print("Final data after stages completed ..\n", self.data)


# Sample Data
# Generate sample customer-level data for 1000 customers over 2 months
np.random.seed(42)  # For reproducibility

# Generate customer IDs and dates
customer_ids = np.arange(1, 1001)
dates = pd.date_range(start="2025-01-01", end="2025-02-28", freq='D')

# Create a DataFrame with all combinations of customer IDs and dates
customer_data = pd.DataFrame({
    "customer_id": np.repeat(customer_ids, len(dates)),
    "date": np.tile(dates, len(customer_ids))
})

# Simulate customer-level features
customer_data["media_channel_clicks"] = np.random.poisson(lam=5, size=len(customer_data))
customer_data["other_channel_clicks"] = np.random.poisson(lam=3, size=len(customer_data))
customer_data["campaign_spend"] = np.random.uniform(10, 100, size=len(customer_data))
customer_data["discounts"] = np.random.choice([0, 10, 20, 30], size=len(customer_data), p=[0.7, 0.1, 0.1, 0.1])
customer_data["leads"] = (customer_data["media_channel_clicks"] + 
                          customer_data["other_channel_clicks"] * 0.8 +
                          np.random.normal(0, 2, len(customer_data))).clip(0).astype(int)
customer_data["sign_ups"] = (customer_data["leads"] * 0.4 +
                             customer_data["campaign_spend"] * 0.05 +
                             np.random.normal(0, 1, len(customer_data))).clip(0).astype(int)
customer_data["revenue"] = (customer_data["sign_ups"] * 100 +
                            customer_data["discounts"] * 50 +
                            np.random.normal(0, 20, len(customer_data))).clip(0).astype(int)


# Execute the Pipeline
config_path = "config.yml"
pipeline = ModelPipeline(customer_data, config_path)
pipeline.run()

