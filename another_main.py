import numpy as np
import pandas as pd

from data_quality import calculate_text_column_metrics, create_report

if __name__ == '__main__':
    # Generate some data
    np.random.seed(0)
    data = {'text': ['Text ' + str(i) for i in range(10)]}
    df = pd.DataFrame(data)

    # Calculate text column metrics
    df, num_unique_words = calculate_text_column_metrics(df)

    # Create report
    report = create_report(df, num_unique_words)
    print(report)
    print(df.text_metrics.iloc[0])