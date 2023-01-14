# from nltk.corpus import webtext, brown
# import numpy as np
# import pandas as pd
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sklearn.model_selection import train_test_split
#
# from cleaning.cleaning import clean_text
# from data_loader.webtext_data import load_data_chat_logs, load_data_pirates, load_data_king_arthur
# from features_engineering.fe_main import get_embeddings
# from mdoel_training.data_preparation import ComplexParameter, CVData, Parameter
# from mdoel_training.logistic_regression import logistic_regression_with_outputs
# from mdoel_training.model_utils import organize_results, analyze_results
# from model_analyzer.model_analysis import analyze_model
# from preprocessing.preprocessing import preprocess_text, get_stemmer
# from quick_analysis.quick_analysis import process_text
# from util import get_stopwords
#
# import matplotlib.pyplot as plt
# import numpy as np
# import numpy as np
# import matplotlib.pyplot as plt
#
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from cleaning.cleaning import clean_text
from data_loader.webtext_data import load_data_pirates, load_data_king_arthur
from features_engineering.fe_main import get_embeddings
from mdoel_training.baseline_model import baseline_with_outputs
from mdoel_training.data_preparation import CVData, Parameter
from mdoel_training.lgbm_class import lgbm_with_outputs
from mdoel_training.lstm import lstm_with_outputs
from mdoel_training.model_results import ModelResults
from mdoel_training.model_utils import organize_results, analyze_results
from model_analyzer.model_analysis import analyze_model
from model_compare.models_comparison import ModelComparison
from preprocessing.preprocessing import preprocess_text, get_stemmer
from util import get_stopwords

#
df1 = load_data_pirates().assign(target='chat_logs')
df2 = load_data_king_arthur().assign(target='pirates')
df = pd.concat([df1, df2])
#
# # Clean the text column
get_sw = get_stopwords()
df['text'] = df['text'].apply(lambda x: clean_text(x,
                                                   remove_stopwords_flag=True,
                                                   stopwords=get_sw))

# preprocess the text column
df['clean_text'] = df['text'].apply(lambda x:
                                    preprocess_text(x, stemmer=get_stemmer('porter'), stem_flag=True))
#
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
#
train_embedding_no_target, test_embedding_no_target = get_embeddings(training_data=train_df, test_data=test_df,
                                                                     corex=True, tfidf=True, bow=True, tfidf_dim=10,
                                                                     bow_dim=10)
#
# # adding the label to train and test embedding
train_df.reset_index(drop=True, inplace=True)
train_embedding_no_target.reset_index(drop=True, inplace=True)
train_embedding = pd.concat([train_embedding_no_target, train_df[['target']]], axis=1)

test_df.reset_index(drop=True, inplace=True)
test_embedding_no_target.reset_index(drop=True, inplace=True)
test_embedding = pd.concat([test_embedding_no_target, test_df[['target']]], axis=1)

# # Set up parameters for logistic regression
# penalty = Parameter('penalty', 'l2')
# C = Parameter('C', 1.0)
# solver = Parameter('solver', 'lbfgs')
# max_iter = Parameter('max_iter', 100)
# #
# parameters = [penalty, C, solver, max_iter]
# # #
# # # # Create CV data
cv_data = CVData(train_data=train_embedding, test_data=test_embedding, folds=3)
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(cv_data.train_data['target'])
label_transformed = label_encoder.transform(cv_data.train_data['target'])
original_labels = label_encoder.inverse_transform(label_transformed)
print(original_labels)

# # #
# # # # Run logistic regression with outputs
# results, model = baseline_with_outputs(cv_data=cv_data, target_col='target', metric_funcs=None)
# results, model, parameters = lgbm_with_outputs(cv_data, [], target_col='target', metric_funcs=None)
# # results, model = logistic_regression_with_outputs(cv_data, parameters, target_col='target', metric_funcs=None)
results, model, parameters, predictions = lstm_with_outputs(cv_data, [], target_col='target', metric_funcs=None,
                                                            label_encoder=label_encoder)

# model_res = ModelResults(model_name="lstm", model=model, results=results, parameters=parameters,
#                          predictions=predictions)
# model_res3 = ModelResults(model_name="lstm", model=model, results=results,
#                           parameters=[Parameter("recurrent_dropout", 0.9)], predictions=predictions)
#
# ModelComparison([model_res, model_res3]).print_top_message()

# # # #
# # # # # Organize results
organized_results = organize_results(results)
print(organized_results.shape)
# # # #
# analyze_results(organized_results, [])
# analyze_model(model, cv_data, target_label='target')
