import numpy as np
import pandas as pd
from nltk.corpus import brown

from data_quality import calculate_text_column_metrics, create_report, plot_text_length_and_num_words, analyze_text_data

if __name__ == '__main__':
    # Generate some data
    np.random.seed(0)
    brown_sent = brown.sents(categories='news')[:100]
    brown_sent = [' '.join(x) for x in brown_sent]
    df = pd.DataFrame({'text': brown_sent})

    # Calculate text column metrics
    df = analyze_text_data(df)

