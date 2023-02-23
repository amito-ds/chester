from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from chester.zero_break.problem_specification import DataInfo


class LinearRegressionModel:
    def __init__(self, parameters: list, data_info: DataInfo):
        self.parameters = parameters
        self.data_info = data_info
        self.model_type = self.data_info.problem_type_val
        if "regression" in self.model_type.lower():
            self.model = ElasticNet()
        self.categorical_features = self.data_info.feature_types_val["categorical"]
        self.numeric_features = self.data_info.feature_types_val["numeric"]
        self.transformer = None
        self.pipeline = None

    def fit(self, X, y):
        import warnings
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
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
        # Fit the pipeline with the updated data
        self.pipeline.fit(X, y)

    def predict(self, X):
        # Use the pipeline to predict the target values
        return self.pipeline.predict(X)

    def transform(self, X):
        return self.predict(X)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def get_params(self):
        return self.parameters
