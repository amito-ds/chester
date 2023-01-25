import pandas as pd

from data_loader.webtext_data import load_data_pirates, load_data_king_arthur
from tcap.features_engineering.feature_main import FeatureExtraction

from tcap.run_full_cycle import run_tcap, DataSpec

df1 = load_data_pirates().assign(target='chat_logs')
df2 = load_data_king_arthur().assign(target='pirates')
df = pd.concat([df1, df2])

origin_df, df, train_embedding, test_embedding, best_model = run_tcap(
    data_spec=DataSpec(df=df, text_column='text', target_column='target'),
    feature_extraction=FeatureExtraction(split_data=False, corex_dim=80, tfidf_dim=90, bow_dim=100),
    is_text_cleaner=False,
    is_text_preprocesser=False,
    is_text_stats=True,
    is_feature_extraction=True,
    is_feature_analysis=False,
    is_train_model=False,
    is_model_analysis=False)

# out = run_tcap(
#     data_spec=DataSpec(df=df, text_column='text', target_column='target'),
#     # is_feature_analysis=True,
#     # is_text_stats=False,
#     # is_feature_extraction=True,
#     # is_train_model=True,
#     # is_model_analysis=True
# )
