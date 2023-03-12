import pandas as pd


def dataframe_to_markdown(df):
    """
    Converts a pandas DataFrame to a Markdown table format.
    """
    df = df.fillna('')
    # Create the table header
    header = "| " + " | ".join(df.columns) + " |"
    separator = "| " + " | ".join(["---" for i in range(len(df.columns))]) + " |"

    # Create the table rows
    rows = []
    for i in range(len(df)):
        row = "| " + " | ".join([str(val) for val in df.iloc[i]]) + " |"
        rows.append(row)

    # Combine the header, separator, and rows into a single string
    markdown_table = "\n".join([header, separator] + rows)

    return markdown_table


# Example usage
df = pd.read_csv("/Users/amitosi/PycharmProjects/chester/projects/project_example.csv")
markdown_table = dataframe_to_markdown(df)
print(markdown_table)
