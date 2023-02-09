import pandas as pd

from chester.features_engineering.fe_nlp import get_embeddings, TextFeatureExtraction
from chester.run.user_classes import TextFeatureSpec


class FeatureHandler:
    def __init__(self, column, col_name,
                 feature_type=None,
                 text_feature_extraction: TextFeatureSpec = None):
        self.column = column
        self.feature_type = feature_type
        self.col_name = col_name
        self.text_feature_extraction = text_feature_extraction

    def handle_numerical(self):
        self.column.name = "num_" + self.column.name
        return self.column, ["num_" + self.col_name]

    def handle_boolean(self):
        column = self.column.astype(int)
        column.name = "is_" + self.column.name
        return column, ["is_" + self.column.name]

    def handle_categorical(self):
        index_col = pd.Series(self.column).astype('category').cat.codes
        index_col.name = "index_" + self.column.name
        # index_col = index_col.replace(-1, 0)
        return index_col, ["index_" + self.column.name]

    def decide_embedding_size(self):
        self.n_unique_words = self.column.nunique()
        self.text_length = self.column.str.len()
        self.n_rows = self.column.shape[0]
        size = None
        if self.n_unique_words < 100 and self.n_rows < 10000:
            size = 30
        elif self.n_unique_words < 1000 and self.n_rows < 100000:
            size = 90
        elif self.n_unique_words < 5000 and self.n_rows < 1000000:
            size = 210
        elif self.n_unique_words < 10000 and self.n_rows < 10000000:
            size = 240
        else:
            size = 300
        return size

    def handle_text(self):
        embedding_size = self.decide_embedding_size()
        print(f"Feature: {self.col_name} calculating {embedding_size} dim embedding")
        embedding_size_method = int(embedding_size / 3)
        data = pd.DataFrame({self.col_name: self.column})

        if self.text_feature_extraction is not None:
            feat_ext = self.text_feature_extraction
            embedding, _ = get_embeddings(
                training_data=data,
                test_data=None,
                split_data=False,
                text_column=self.col_name,
                corex_dim=feat_ext.corex_dim, corex=feat_ext.corex,
                bow_dim=feat_ext.bow_dim, bow=feat_ext.bow,
                tfidf_dim=feat_ext.tfidf_dim, tfidf=feat_ext.tfidf,
                ngram_range=feat_ext.ngram_range
            )

        else:
            embedding, _ = get_embeddings(training_data=data, split_data=False, text_column=self.col_name,
                                          corex_dim=embedding_size_method, bow_dim=embedding_size_method,
                                          tfidf_dim=embedding_size_method)

        new_col_name_list = [self.col_name + "_" + col for col in embedding.columns]
        new_col_name_dict = dict(zip(embedding.columns, new_col_name_list))
        embedding.rename(columns=new_col_name_dict, inplace=True)
        return embedding, embedding.columns

    def handle_feature(self):
        if self.feature_type == 'numeric':
            return self.handle_numerical()
        elif self.feature_type == 'categorical':
            return self.handle_categorical()
        elif self.feature_type == 'boolean':
            return self.handle_boolean()
        elif self.feature_type == 'text':
            return self.handle_text()
        elif self.feature_type == 'time':
            return None, None
        else:
            if not (self.col_name == self.col_name):
                print(
                    f"No appropriate feature handler found for feature {self.col_name}. This feature will be ignored.")
                return None, None

# Example usage:
# df = pd.DataFrame({'age': [21, 22, 23], 'name': ['Alice', 'Bob', 'Charlie']})
# col = df['age']
# col_name = 'age'
# feature_type = 'numerical'
#
# fh = FeatureHandler(col, feature_type, col_name)
# print(fh.handle_numerical())

# df = pd.DataFrame({'A': [1, 2, 3], 'B': [True, False, True], 'C': ['a', 'b', 'c'], 'D': ['apple', 'banana', 'orange']})
# column = df['C']
# feature_type = 'boolean'
#
# feature_handler = FeatureHandler(column, feature_type, 'D')
# a, b = feature_handler.handle_categorical()
# print(b)
# print(a)
#
# df1 = load_data_pirates().assign(target='chat_logs')
# df2 = load_data_king_arthur().assign(target='pirates')
# df = pd.concat([df1, df2])
# feature_handler = FeatureHandler(df['text'], feature_type='categorical', col_name='text')


# df = pd.DataFrame({'A': [1, 2, 3], 'B': [True, False, True], 'C': ['a', 'b', 'c'], 'D': ['apple', 'banana', 'orange']})
# column = df['B']
# feature_type = 'boolean'
# feature_handler = FeatureHandler(column, feature_type=feature_type, col_name='C')
# a, b = feature_handler.handle_feature()
# print(b)
# print(a[0:3])
