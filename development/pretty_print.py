from typing import List

from IPython.display import display
from IPython.core.display import HTML
import pandas as pd


def print_dataframe(df: pd.DataFrame, n_rows: int = None, trim_text_cols: int = None, decimal_places: int = 2,
                    color_cols: list = None):
    """
    Prints a dataframe in a user-friendly format
    :param df: The dataframe to print
    :param n_rows: The number of rows to print. If None, all rows will be printed
    :param trim_text_cols: The number of characters to trim text columns by. If None, no trimming will be done
    :param decimal_places: The number of decimal places to round numeric columns to.
    :param color_cols: A list of tuples, each containing a column name from the dataframe and a color.
    """
    if n_rows is not None:
        df = df.head(n_rows)
    if trim_text_cols is not None:
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str[:trim_text_cols]
    if decimal_places is not None:
        for col in df.select_dtypes(include=['float']).columns:
            df[col] = df[col].round(decimal_places)
    if color_cols is not None:
        for col, color in color_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"<span style='color: {color}'>{x}</span>")
    display(HTML(df.to_html()))


import pandas as pd

data = {
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'age': [25, 32, 45, 27],
    'gender': ['F', 'M', 'M', 'M'],
    'color': ['red', 'green', 'blue', 'yellow'],
    'score': [8.5, 7.3, 9.0, 6.5]
}

# df = pd.DataFrame(data)
#
# # Print the first 5 rows of the dataframe
# print_dataframe(df, n_rows=5)
#
# # Print the first 5 rows of the dataframe, trimming text columns by 20 characters, rounding numeric columns to 2 decimal places
# print_dataframe(df, n_rows=5, trim_text_cols=20, decimal_places=2)
#
# # Print the first 5 rows of the dataframe, with certain columns highlighted in specific colors
# color_cols = [('column_name', 'red'), ('another_column', 'blue')]
# print_dataframe(df, n_rows=5, color_cols=color_cols)

import matplotlib.pyplot as plt
import pandas as pd
import random

import matplotlib.pyplot as plt
import pandas as pd
import random
import matplotlib.pyplot as plt
import pandas as pd
import random


def format_data(df: pd.DataFrame, decimal_points=2, max_chars=20):
    df_formatted = df.copy()
    for col in df.columns:
        if df[col].dtype == 'float':
            df_formatted[col] = df[col].round(decimal_points)
        elif df[col].dtype == 'object':
            df_formatted[col] = df[col].str[:max_chars]
    return df_formatted


def plot_words(df_formatted: pd.DataFrame, font_size=None):
    # Create a list of colors for the words
    colors = [random.choice(['red', 'blue', 'green', 'purple', 'black', 'orange']) for i in
              range(df_formatted.shape[1])]
    # Create an empty figure and axis
    fig, ax = plt.subplots(figsize=(10, 20))
    # Iterate through the columns
    for i, col in enumerate(df_formatted.columns):
        # Plot each word in the specified color
        for j, word in enumerate(df_formatted[col]):
            if font_size is None:
                font_size = calculate_font_size(df_formatted)
            ax.text(i + 1, len(df_formatted) - j, word, color=colors[i], fontsize=font_size, ha='center')
    plt.xlim(0, len(df_formatted.columns) + 1)
    plt.ylim(0, len(df_formatted) + 1)
    ax.set_frame_on(False)
    plt.show()


def calculate_font_size(df_formatted: pd.DataFrame):
    # Get the width and height of the plot
    width, height = plt.gcf().get_size_inches()
    # Calculate the average length of the words in the dataframe
    avg_word_length = df_formatted.apply(lambda x: x.str.len()).mean().mean()
    # Get the maximum length of the column names
    max_col_name_length = max(len(col) for col in df_formatted.columns)
    # Calculate the optimal font size based on the size of the dataframe, the length of the words, and the size of the plot
    font_size = (width * height) / ((len(df_formatted) + max_col_name_length) * avg_word_length)
    return font_size


# your code to calculate font size based on data size and other parameters


import pandas as pd


def print_word_with_color(value, color: str):
    color_codes = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "purple": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "black": "\033[30m"
    }
    if color in color_codes:
        print(color_codes[color] + str(value) + "\033[0m")
    else:
        print(value)


from tabulate import tabulate


def print_pandas_with_colors(df: pd.DataFrame):
    for i, row in df.iterrows():
        for j, col in enumerate(df.columns):
            if not col.lower().endswith('color'):
                if j == 0:
                    print_word_with_color(row[col], 'white')
                else:
                    color = row[col + ' Color']
                    print_word_with_color(row[col], color)
        print()



# Create a sample DataFrame
# Example 1
import seaborn as sns
import pandas as pd

# Create a sample dataset
data = {'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8], 'C': [9, 10, 11, 12]}
df = pd.DataFrame(data)

# Create the heatmap
sns.heatmap(df, cmap='coolwarm')

# Show the plot
plt.show()
