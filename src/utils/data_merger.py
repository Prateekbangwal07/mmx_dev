import pandas as pd
import yaml
from typing import List, Dict

class DataFrameMerger:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.base_df = None
        self.target_granularity = None

    def set_granularity_columns(self, df: pd.DataFrame, date_column: str):
        """
        Add granularity ID columns (DAY_ID, WEEK_ID, etc.) based on the date column.
        """
        df['DAY_ID'] = pd.to_datetime(df[date_column])
        df['WEEK_ID'] = df['DAY_ID'].dt.to_period('W').apply(lambda r: r.start_time)
        df['MONTH_ID'] = df['DAY_ID'].dt.to_period('M').apply(lambda r: r.start_time)
        df['QUARTER_ID'] = df['DAY_ID'].dt.to_period('Q').apply(lambda r: r.start_time)
        df['YEAR_ID'] = df['DAY_ID'].dt.year

    def adjust_to_target_granularity(
    self, df: pd.DataFrame, agg_func: str, value_columns: List[str], merge_columns: List[str]):
        """
        Adjust the DataFrame to the specified granularity using aggregation functions.
        """

        # Add granularity column to merge columns if not already present
        if self.target_granularity not in merge_columns:
            merge_columns.append(self.target_granularity)

        # Define aggregation functions
        agg_funcs = {col: agg_func for col in value_columns}
        return df.groupby(merge_columns).agg(agg_funcs).reset_index()


    def merge_dataframes(self):
        """
        Merge all dataframes into the base dataframe.
        """
        for df_config_name, df_config in self.config['dataframes'].items():
            if df_config.get('base_df_flag', False):
                continue  # Skip the base dataframe during merging

            df = pd.read_csv(f"data/{df_config_name}.csv")
            self.set_granularity_columns(df, df_config['date_column'])
            
            # Adjust granularity to match the target
            df_adjusted = self.adjust_to_target_granularity(
                df,
                df_config['aggregation'],
                df_config['value_columns'],
                df_config['merge_columns']  # Pass merge columns here
            )

            # Merge using columns from config
            merge_columns = df_config['merge_columns']
            self.base_df = pd.merge(
                self.base_df, df_adjusted, on=merge_columns, how='left'
            )

    def process(self):
        """
        Load the base dataframe, set its granularity, and merge other dataframes.
        """
        for df_config_name, df_config in self.config['dataframes'].items():
            if df_config.get('base_df_flag', False):
                self.base_df = pd.read_csv(f"data/{df_config_name}.csv")
                self.set_granularity_columns(self.base_df, df_config['date_column'])
                self.target_granularity = df_config['granularity'] + '_ID'
                break

        self.merge_dataframes()
        return self.base_df


if __name__ == "__main__":
    merger = DataFrameMerger('config.yml')
    merged_df = merger.process()
    print(merged_df)
