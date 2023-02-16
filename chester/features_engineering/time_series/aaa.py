import numpy as np
import pandas as pd

df = pd.read_csv("/Users/amitosi/PycharmProjects/chester/chester/data/day.csv")
df.rename(columns={'cnt': 'target'}, inplace=True)

date_col = 'dteday'
target_col = 'target'

# Sort the DataFrame by the date column in ascending order
df = df.sort_values(date_col)

# Use rolling window with a size of 10 to collect the latest 10 values of the target by date for each row
df['last_10_targets'] = df[target_col].rolling(window=10).apply(lambda x: x.shift().tail(10)).apply(
    lambda x: list(x.dropna()), axis=1)

# Reset the index to the default integer index
df.reset_index(inplace=True)
