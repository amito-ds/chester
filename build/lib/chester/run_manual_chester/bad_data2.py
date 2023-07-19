import pandas as pd
import os

# Load the data
data_path = "/Users/amitosi/PycharmProjects/chester/chester/run_manual_chester/bbc.csv"
bbc_df = pd.read_csv(data_path)

# Remove the unnecessary column and check data types
bbc_df = bbc_df.drop(columns=["Unnamed: 0"])
print("Data types:")
print(bbc_df.dtypes)

# Print the first few rows of data
print("\nSample rows:")
print(bbc_df.head())

# Perform EDA using the provided code snippet
from chester.run import full_run as fr
from chester.run import user_classes as uc

eda_results = fr.run(uc.Data(df=bbc_df))
print("\nEDA Results:")
print(eda_results)