import pandas as pd
import yaml
import os

class SalesDataProcessor:
    def __init__(self, config_path: str, input_file: str, output_file: str):
        """
        Initializes the SalesDataProcessor with configuration, input, and output paths.

        Args:
            config_path (str): Path to the YAML configuration file.
            input_file (str): Path to the input CSV file.
            output_file (str): Path for the processed output CSV file.
        """
        self.config_path = config_path
        self.input_file = input_file
        self.output_file = output_file
        self.config = self._load_config()

    def _load_config(self):
        """Loads and parses the YAML configuration file."""
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)

    def process_data(self):
        """Processes the sales data based on the configuration."""
        # Load the CSV file
        df = pd.read_csv(self.input_file)
        df = df[df['Price']>0]

        # Fetch configurations
        preprocessing_config = self.config['preprocessing']['sales_pre_processing']
        selected_columns = preprocessing_config['selected_columns']
        merged_columns = preprocessing_config['merged_columns']
        value_columns = preprocessing_config['value_columns']
        value_agg_columns = preprocessing_config['value_agg_columns']

        # Step 1: Filter the selected columns
        df = df[selected_columns]

        # Step 2: Group by merged columns and aggregate
        agg_dict = {col: func for col, func in zip(value_columns, value_agg_columns)}
        grouped_df = df.groupby(merged_columns).agg(agg_dict).reset_index()

        # Step 3: Save the processed DataFrame to the output file
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        grouped_df.to_csv(self.output_file, index=False)

        print(f"Processed data saved to {self.output_file}")

# Example usage
if __name__ == "__main__":
    processor = SalesDataProcessor(
        config_path="config.yml",
        input_file="data/sales.csv",
        output_file="data/output/bronze_level_sales.csv"
    )
    processor.process_data()
