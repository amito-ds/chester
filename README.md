# TCAP
TCAP is a Python package for natural language 
processing tasks, including text cleaning, 
pre-processing, text stats,  feature extraction, 
feature analysis, model training, 
and post-model analysis. All in 1 line of code, given the data.

Installation
You can install TCAP using pip:

pip install TCAP

# Usage

To use TCAP, you'll need a dataframe containing a 
text column (named 'text') and a target column (named 'target'). 
Here's an example of how to use the run_tcap function:

````python
import pandas as pd

from data_loader.webtext_data import load_data_pirates, load_data_king_arthur
from tcap.run_full_cycle import run_tcap, DataSpec

df1 = load_data_pirates().assign(target='chat_logs')
df2 = load_data_king_arthur().assign(target='pirates')
df = pd.concat([df1, df2])
origin_df, df, train_embedding, test_embedding, best_model = run_tcap(
    data_spec=DataSpec(df=df, text_column='text', target_column='target'),
)
````

### This will:

perform text cleaning and pre-processing (some but not limited to: lower case, remove punctuation, stemming, remove stopwords...)
Text statistic. Some but not limited to: text length, # of words, common words, word cloud, topic analysis (corex, LSA), and more
feature extraction: will create concatenate embedding using tf-idf, bow and corex
Pre-model analysis: some but not limited to: will perform feature analysis to the label. will calculate correlation matrix to the label, calculate chi square for each feature VS the label, will perform LSA to dimensionality reduction and then see if the new shape explains the classes, and more
training a model. it will choose between logistic regression and lgbm with some best practice choice of parameters
post-model analysis: the best model will get analyzed and calculate performance, confusion matrix, accuracy, shap, feature importance, and more
More logic to follow in the near future

# Contributing
If you are interested in contributing to TCAP, please see our CONTRIBUTING guidelines.

# License
TCAP is released under the MIT License.

# Acknowledgements
The TCAP package was developed by Amit Osi
TCAP makes use of the following open-source libraries: NLTK, spaCy, Gensim, Sklearn.

