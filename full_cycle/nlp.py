from typing import Any, Dict

import pandas as pd

from cleaning.cleaning import clean_text
from data_loader.webtext_data import load_data_pirates, load_data_king_arthur
from feature_analyzing.feature_correlation import PreModelAnalysis
from features_engineering.fe_main import get_embeddings
from preprocessing.preprocessing import preprocess_text, get_stemmer
from text_analyzer.smart_text_analyzer import analyze_text
from util import get_stopwords


class NLP:
    def __init__(self, df: pd.DataFrame, text_column: str = 'text', target_column: str = 'target'):
        self.df = df
        self.text_column = text_column
        self.target_column = target_column

    def run_cleaning_text(self, cleaning_params: Dict[str, Any] = None):
        """
        Perform text cleaning on the text_column of the dataframe using the provided parameters
        """
        if not cleaning_params:
            cleaning_params = {}
        remove_punctuation_flag = cleaning_params.get('remove_punctuation_flag', True)
        remove_numbers_flag = cleaning_params.get('remove_numbers_flag', True)
        remove_whitespace_flag = cleaning_params.get('remove_whitespace_flag', True)
        remove_empty_line_flag = cleaning_params.get('remove_empty_line_flag', True)
        lowercase_flag = cleaning_params.get('lowercase_flag', True)
        remove_stopwords_flag = cleaning_params.get('remove_stopwords_flag', True)
        stopwords = cleaning_params.get('stopwords', None)
        remove_accented_characters_flag = cleaning_params.get('remove_accented_characters_flag', True)
        remove_special_characters_flag = cleaning_params.get('remove_special_characters_flag', True)
        remove_html_tags_flag = cleaning_params.get('remove_html_tags_flag', True)

        self.df[self.text_column] = self.df[self.text_column].apply(lambda x: clean_text(x,
                                                                                         remove_punctuation_flag,
                                                                                         remove_numbers_flag,
                                                                                         remove_whitespace_flag,
                                                                                         remove_empty_line_flag,
                                                                                         lowercase_flag,
                                                                                         remove_stopwords_flag,
                                                                                         stopwords,
                                                                                         remove_accented_characters_flag,
                                                                                         remove_special_characters_flag,
                                                                                         remove_html_tags_flag))
        return self.df

    def run_preprocessing(self, preprocessing_params: Dict[str, Any] = None):
        """
        Perform preprocessing on the text_column of the dataframe using the provided parameters
        """
        if not preprocessing_params:
            preprocessing_params = {}
        stemmer = preprocessing_params.get('stemmer', get_stemmer())
        lemmatizer = preprocessing_params.get('lemmatizer', None)
        stem_flag = preprocessing_params.get('stem_flag', False)
        lemmatize_flag = preprocessing_params.get('lemmatize_flag', False)
        tokenize_flag = preprocessing_params.get('tokenize_flag', True)
        pos_tag_flag = preprocessing_params.get('pos_tag_flag', False)

        self.df['clean_text'] = self.df[self.text_column].apply(lambda x: preprocess_text(x,
                                                                                          stemmer=stemmer,
                                                                                          lemmatizer=lemmatizer,
                                                                                          stem_flag=stem_flag,
                                                                                          lemmatize_flag=lemmatize_flag,
                                                                                          tokenize_flag=tokenize_flag,
                                                                                          pos_tag_flag=pos_tag_flag))
        return self.df

    def run_text_analyze(self, text_analyze_params: Dict[str, Any] = None):
        """
        Perform text analysis on the text_column of the dataframe using the provided parameters
        """
        if not text_analyze_params:
            text_analyze_params = {}
        create_wordcloud = text_analyze_params.get('create_wordcloud', True)
        corex_topics = text_analyze_params.get('corex_topics', True)
        key_sentences = text_analyze_params.get('key_sentences', True)
        common_words = text_analyze_params.get('common_words', True)
        sentiment = text_analyze_params.get('sentiment', True)
        data_quality = text_analyze_params.get('data_quality', True)
        corex_topics_num = text_analyze_params.get('corex_topics_num', 10)
        top_words = text_analyze_params.get('top_words', 10)
        n_sentences = text_analyze_params.get('n_sentences', 5)

        analyze_text(df=self.df,
                     create_wordcloud=create_wordcloud,
                     corex_topics=corex_topics,
                     key_sentences=key_sentences,
                     common_words=common_words,
                     sentiment=sentiment,
                     data_quality=data_quality,
                     corex_topics_num=corex_topics_num,
                     top_words=top_words,
                     n_sentences=n_sentences)

    # def run_feature_extraction_engineering(self, feature_extraction_engineering_params: Dict[str, Any] = None):
    #     """
    #     Perform model pre-analysis on the text_column of the dataframe using the provided parameters
    #     """
    #     if not feature_extraction_engineering_params:
    #         feature_extraction_engineering_params = {}
    #     corex = feature_extraction_engineering_params.get('corex', True)
    #     corex_dim = feature_extraction_engineering_params.get('corex_dim', 100)
    #     tfidf = feature_extraction_engineering_params.get('tfidf', True)
    #     tfidf_dim = feature_extraction_engineering_params.get('tfidf_dim', 10000)
    #     bow = feature_extraction_engineering_params.get('bow', True)
    #     bow_dim = feature_extraction_engineering_params.get('bow_dim', 10000)
    #     ngram_range = feature_extraction_engineering_params.get('ngram_range', (1, 1))
    #     split_data = feature_extraction_engineering_params.get('split_data', True)
    #     split_prop = feature_extraction_engineering_params.get('split_prop', 0.3)
    #     split_random_state = feature_extraction_engineering_params.get('split_random_state', 42)
    #     text_column = feature_extraction_engineering_params.get('text_column', 'clean_text')
    #     target_col = feature_extraction_engineering_params.get('target_col', 'target')
    #     self.embeddings, self.test_embeddings = get_embeddings(self.df, text_column=text_column, target_col=target_col,
    #                                                            split_data=split_data, split_prop=split_prop,
    #                                                            split_random_state=split_random_state,
    #                                                            corex=corex, corex_dim=corex_dim, tfidf=tfidf,
    #                                                            tfidf_dim=tfidf_dim, bow=bow, bow_dim=bow_dim,
    #                                                            ngram_range=ngram_range)

    def run_model_pre_analysis(self, model_pre_analysis_params: Dict[str, Any] = None):
        """
        Perform pre-model analysis on the dataframe using the provided parameters
        """
        if not model_pre_analysis_params:
            model_pre_analysis_params = {}
        correlation_matrix = model_pre_analysis_params.get('correlation_matrix', True)
        tsne_plot = model_pre_analysis_params.get('tsne_plot', True)
        top_n_pairplot = model_pre_analysis_params.get('top_n_pairplot', True)
        chi_square_test_all_features = model_pre_analysis_params.get('chi_square_test_all_features', True)
        top_n_features = model_pre_analysis_params.get('top_n_features', 200)

        print(self.df[0:10])
        pre_model_analysis = PreModelAnalysis(df=self.df, target_column=self.target_column,
                                              top_n_features=top_n_features)
        pre_model_analysis.run(correlation_matrix=correlation_matrix, tsne_plot=tsne_plot,
                               top_n_pairplot=top_n_pairplot,
                               chi_square_test_all_features=chi_square_test_all_features)

