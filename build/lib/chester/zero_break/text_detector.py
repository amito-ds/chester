import pandas as pd


def determine_if_text_or_categorical_column(column):
    # Check if column is a string or categorical data type
    if column.dtype == 'object' or column.dtype.name == 'category':
        # less than 20 unique values -> not a text column
        if len(column.unique()) < 20:
            return False, True
        # Check if column contains more than 50% unique values
        elif len(column.unique()) > 0.5 * len(column):
            return True, False  # Text column
        else:
            return False, True  # Categorical column
    return False, False  # Not a text or categorical column


def test_determine_if_text_column():
    data = {'col1': ['this is text', 'so is this', 'this is not', 1, 2, 3]}
    df = pd.DataFrame(data)
    assert (determine_if_text_or_categorical_column(df['col1']) == (True, False))
    data = {'col1': [1, 2, 3, 4, 5, 6]}
    df = pd.DataFrame(data)
    assert (determine_if_text_or_categorical_column(df['col1']) == (False, False))
    data = {'col1': ['cat', 'dog', 'cat', 'dog', 'cat', 'dog', 'cat', 'dog', 'cat', 'dog']}
    df = pd.DataFrame(data)
    assert (determine_if_text_or_categorical_column(df['col1']) == (False, True))
    import random
    data = {'col1': [random.choice(
        ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
         'w', 'x', 'y', 'z']) for _ in range(1000)]}
    df = pd.DataFrame(data)
    assert (determine_if_text_or_categorical_column(df['col1']) == (False, True))
    data = {'col1': ['cat', 'dog', 'cat', 'dog']}
    df = pd.DataFrame(data)
    assert (determine_if_text_or_categorical_column(df['col1']) == (False, True))
