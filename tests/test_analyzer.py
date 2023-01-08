import pandas as pd

from data_quality import calculate_text_statistics


def test_calculate_text_statistics():
    df = pd.DataFrame({'text': ['this is a test', 'this is another test']})
    report = calculate_text_statistics(df)
    assert "Number of rows with missing data: 0" in report
    assert "Number of unique words: 5" in report
    assert "Average number of words per text: 4.00" in report
    assert "Average number of sentences per text: 1.00" in report
    assert "Average length of text: 17.00" in report


test_calculate_text_statistics()
