import numpy as np
from scipy.optimize._linesearch import LineSearchWarning
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from chester.zero_break.problem_specification import DataInfo


class LogisticRegressionModel:
    def __init__(self, parameters: list, data_info: DataInfo):
        self.parameters = parameters
        self.model = LogisticRegression()
        self.data_info = data_info
        self.categorical_features = self.data_info.feature_types_val["categorical"]
        self.numeric_features = self.data_info.feature_types_val["numeric"]
        self.transformer = None
        self.pipeline = None
        self.feature_names = None

    def fit(self, X, y):
        import warnings
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=LineSearchWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        # Prepare the numerical features for processing
        numerical_transformer = SimpleImputer(strategy='median')
        # Prepare the categorical features for processing
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        # Use the ColumnTransformer to apply the transformations to the corresponding features
        self.transformer = ColumnTransformer(
            transformers=[
                ('c', categorical_transformer, self.categorical_features),
                ('n', numerical_transformer, self.numeric_features)
            ])
        # Prepare the pipeline
        self.pipeline = Pipeline(steps=[('preprocessor', self.transformer),
                                        ('regressor', self.model)
                                        ])

        # Fit the pipeline with the training data
        hyperparams = {param.name: param.value for param in self.parameters}
        self.model.set_params(**hyperparams)
        self.pipeline.fit(X, y)

    def retrain(self, X, y):
        self.pipeline.fit(X, y)

    def transform(self, X):
        return self.pipeline.predict_proba(X)

    def predict(self, X):
        return self.pipeline.predict(X)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def get_params(self):
        return self.parameters

    def coef_(self):
        return self.model.coef_
