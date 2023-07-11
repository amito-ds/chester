def round_value(x):
    try:
        if abs(x) < 10:
            return round(x, 2)
        elif abs(x) < 100:
            return round(x, 1)
        else:
            return round(x)
    except:
        return None


from prettytable import PrettyTable


def create_pretty_table(df):
    # Convert the DataFrame to a list of dictionaries
    data = df.reset_index(drop=True).to_dict('records')

    # Create a PrettyTable object with column names
    table = PrettyTable(df.columns.tolist())

    # Add rows to the table
    for row in data:
        table.add_row([row[column] for column in df.columns])

    # Return the table
    return table
