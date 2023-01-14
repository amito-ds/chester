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

df = pd.DataFrame(data)

# Print the first 5 rows of the dataframe
print_dataframe(df, n_rows=5)

# Print the first 5 rows of the dataframe, trimming text columns by 20 characters, rounding numeric columns to 2 decimal places
print_dataframe(df, n_rows=5, trim_text_cols=20, decimal_places=2)

# Print the first 5 rows of the dataframe, with certain columns highlighted in specific colors
color_cols = [('column_name', 'red'), ('another_column', 'blue')]
print_dataframe(df, n_rows=5, color_cols=color_cols)
