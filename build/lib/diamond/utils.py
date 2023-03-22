import pandas as pd
import numpy as np


def index_labels(y):
    # Calculate unique values of y and sort them
    unique_values = get_unique_values(y)
    # Create a dictionary with the unique values as keys and their corresponding index as values
    label_dict = {val: i for i, val in enumerate(unique_values)}
    # Create a new Pandas Series with the indexed values of y
    y_indexed = pd.Series([label_dict[val] for val in y])
    # Return the label dictionary and the indexed Series
    return label_dict, y_indexed


def get_unique_values(y):
    try:
        unique = y.unique()
    except:
        unique = np.unique(np.array(y))
    return sorted(unique)
