import warnings

from tcap.model_analyzer.model_analysis import analyze_model

from tcap.text_analyzer.smart_text_analyzer import analyze_text_df

from tcap.chapter_messages import chapter_message
from tcap.text_analyzer.data_quality import TextAnalyzer

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
import logging

logging.basicConfig(level=logging.WARNING)

from tcap.model_training.data_preparation import CVData
from tcap.model_training.model_utils import analyze_results
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tcap.cleaning import cleaning_func as cln
from tcap.preprocessing import preprocessing_func as pp
from tcap.features_engineering import feature_main as fe_main
from tcap.feature_analyzing import feature_correlation
from tcap.model_training import data_preparation
from tcap.model_training import best_model as bm


def parameter_completer(instance1, instance2):
    attributes1 = vars(instance1)
    attributes2 = vars(instance2)
    for key in attributes1:
        if key in attributes2:
            setattr(instance2, key, getattr(instance1, key))


def parameter_super_completer(instance_list: list, instance):
    for instance_i in instance_list:
        parameter_completer(instance_i, instance)


class DataSpec:
    def __init__(self, df: pd.DataFrame, text_column: str = 'text', target_column: str = None):
        self.df = df
        self.text_column = text_column
        self.target_column = target_column


def run_tcap(
        data_spec: DataSpec = None,
        text_cleaner: cln.TextCleaner = None, is_text_cleaner: bool = True,
        text_preprocesser: pp.TextPreprocessor = None, is_text_preprocesser: bool = True,
        text_analyzer: TextAnalyzer = None, is_text_stats: bool = True,
        feature_extraction: fe_main.FeatureExtraction = None, is_feature_extraction: bool = True,
        feature_analysis: feature_correlation.PreModelAnalysis = None, is_feature_analysis: bool = True,
        cv_data: data_preparation.CVData = None,
        model_cycle: bm.ModelCycle = None, model_compare: bm.CompareModels = None,
        is_train_model: bool = True,
        is_model_analysis: bool = True
):
    """
    This function runs the full data processing and model training pipeline. It takes a series of optional inputs
    (text_cleaner, text_preprocesser, text_analyzer, feature_extraction, feature_analysis, cv_data, model_cycle)
    and applies them to the data according to the corresponding flags (is_text_cleaner, is_text_preprocesser,
    is_text_stats, is_feature_extraction, is_feature_analysis, is_train_model, is_model_analysis). The function also
    takes an optional input 'data_spec' which can be used to pass any additional information required for the pipeline.
    The function returns the cleaned and preprocessed dataframe after all the processing steps have been applied.

    Parameters:
    data_spec (DataSpec, optional): An object containing the data, text column and (optional) target column
    text_cleaner (cleaning.cleaning.TextCleaner, optional): An object containing the text cleaning settings.
    is_text_cleaner (bool, optional): A flag indicating whether text cleaning should be applied.
    text_preprocesser (preprocessing.preprocessing.TextPreprocessor, optional): An object containing the text preprocessing settings.
    is_text_preprocesser (bool, optional): A flag indicating whether text preprocessing should be applied.
    text_analyzer (TextAnalyzer, optional): An object containing the text analysis settings.
    is_text_stats (bool, optional): A flag indicating whether text statistics should be generated.
    feature_extraction (features_engineering.fe_main.FeatureExtraction, optional): An object containing the feature extraction settings.
    is_feature_extraction (bool, optional): A flag indicating whether feature extraction should be applied.
    feature_analysis (feature_analyzing.feature_correlation.PreModelAnalysis, optional): An object containing the feature analysis settings.
    is_feature_analysis (bool, optional): A flag indicating whether feature analysis should be applied.
    cv_data (model_training.data_preparation.CVData, optional): An object containing the data for cross-validation.
    model_cycle (model_training.best_model.ModelCycle, optional): An object containing the model training settings.
    is_train_model (bool, optional): A flag indicating whether model training should be applied.
    is_model_analysis (bool, optional): A flag indicating whether model analysis should be applied.

    Actions:
    perform requested tasks

    """

    # Step 0: prepare outputs
    df, train_embedding, test_embedding = None, None, None
    print(chapter_message(chapter_name="Creating report using TCAP:", prefix=""))
    report = ""
    report += "We're getting ready to start our data journey with TCAP.\n"
    if is_text_cleaner:
        report += "First, we'll be cleaning the text data to ensure that it is free of any unnecessary information. This will make it ready for further analysis.\n"
    if is_text_preprocesser:
        report += "Next, we'll be preprocessing the text data so that it's in the right format for feature extraction.\n"
    if is_text_stats:
        report += "With the text data prepped and ready, we'll dive deeper by analyzing it to understand its characteristics and generate statistics.\n"
    if is_feature_extraction:
        report += "With a good understanding of the text data, we'll move on to extracting features from it. This will allow us to represent the text data in a numerical format.\n"
    if is_feature_analysis:
        report += "Before training the model, we'll analyze the extracted features to understand their relationship with the target variable.\n"
    if is_train_model:
        report += "With all the preparation done, we'll now move on to training the model using the extracted features and evaluate its performance.\n"
    if is_model_analysis:
        report += "Finally, we'll conduct a model analysis to understand the model's performance and behavior.\n"
    print(report)

    # Step 1: Prepare text_cleaner object
    if not text_cleaner:
        text_cleaner = cln.TextCleaner()
    if data_spec:
        try:
            parameter_completer(data_spec, text_cleaner)
        except:
            pass
    origin_df = text_cleaner.df.copy()

    # Step 2: Apply Text Cleaning
    if is_text_cleaner:
        print(chapter_message("cleaning"))
        text_cleaner.generate_report()
        df = cln.clean_text_df(text_cleaner)

    # Step 3: Prepare text_preprocesser object
    if is_text_preprocesser:
        if not text_preprocesser:
            text_preprocesser = pp.TextPreprocessor()
        try:
            parameter_completer(text_cleaner, text_preprocesser)
        except:
            pass
        # pp
        print(chapter_message("preprocessing"))
        text_preprocesser.generate_report()
        df = pp.preprocess_text_df(text_preprocesser)

    # Step 4: Prepare text_analyzer object
    if is_text_stats:
        if not text_analyzer:
            text_analyzer = TextAnalyzer()
        try:
            parameter_completer(text_cleaner, text_analyzer)
        except:
            pass
        try:
            parameter_completer(text_preprocesser, text_analyzer)
        except:
            pass
        print(chapter_message("text analyze"))
        analyze_text_df(text_analyzer)

    # Step 5: Feature extraction
    if is_feature_extraction:
        if not feature_extraction:
            feature_extraction = fe_main.FeatureExtraction(training_data=df)
        if feature_extraction.training_data is None:
            feature_extraction.training_data = df
        try:
            parameter_super_completer([text_cleaner, text_preprocesser], feature_extraction)
        except:
            pass
        print(chapter_message("create embedding"))
        train_embedding, test_embedding = fe_main.extract_features(feature_extraction)

    # special case: not found target column
    if 'target' not in df.columns:
        is_feature_analysis = False
        is_train_model = False
        is_model_analysis = False

    # Step 6: Feature analysis and preparation for model training
    if is_feature_analysis or is_model_analysis or is_train_model:
        try:
            target_column = feature_extraction.target_column
        except:
            target_column = 'target'
        label_encoder = LabelEncoder()
        train_embedding[target_column] = label_encoder.fit_transform(train_embedding[target_column])
        test_embedding[target_column] = label_encoder.transform(test_embedding[target_column])

    # Pre model analysis
    if is_feature_analysis:
        if not feature_analysis:
            feature_analysis = feature_correlation.PreModelAnalysis(df=train_embedding,
                                                                    target_column=target_column)
        print(chapter_message("model pre analysis"))
        feature_analysis.generate_report()
        feature_analysis.run()

    # Step 7: Model training
    best_model = None
    if is_train_model:
        print(chapter_message("model run"))
        if not cv_data:
            cv_data = CVData(train_data=train_embedding, test_data=test_embedding)
        if not model_cycle:
            model_cycle = bm.ModelCycle(cv_data=cv_data, target_col=target_column,
                                        logistic_regression_models=10, lgbm_models=1)
        if model_compare is not None:
            model_cycle.compare_models = model_compare
        best_model = model_cycle.get_best_model()

    # Step 8: Model analysis
    if is_model_analysis:
        print(chapter_message("post model analysis"))
        organized_results = pd.DataFrame(best_model.results)
        analyze_results(organized_results, best_model.parameters)
        analyze_model(best_model.model, cv_data, target_label=target_column)
    return origin_df, df, train_embedding, test_embedding, best_model
