import pandas as pd


def format_df(df, max_cols_width=10, max_value_width=10):
    # pd.options.display.max_colwidth = max_cols_width
    pd.options.display.max_columns = None

    def trim_value(val):
        if len(str(val)) > max_value_width:
            return str(val)[:max_value_width] + "..."
        return str(val)

    df = df.applymap(trim_value)
    print(df)
    return df
    # return df.to_string(index=False, columns=False)
