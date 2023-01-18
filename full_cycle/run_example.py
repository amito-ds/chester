import pandas as pd

from data_loader.webtext_data import load_data_pirates, load_data_king_arthur
from full_cycle.nlp import NLP

# TO DO: add warning message when using sentiment with removing stop words
# TO DO: add warning message when using cleaning and pp to present key sentences and LSA
# TO DO: make sense of the FE + analyzing

df1 = load_data_pirates().assign(target='chat_logs')
df2 = load_data_king_arthur().assign(target='pirates')
df = pd.concat([df1, df2])

nlp = NLP(df, text_column='text', target_column='target')
# nlp.run_cleaning_text()
# nlp.run_preprocessing()
# print(nlp.df[0:10])
# # print(nlp.df[0:10])
# nlp.run_text_analyze()
# nlp.run_feature_extraction_engineering()
# nlp.run_model_pre_analysis()
# sw = get_stopwords()
# print('not' in sw)