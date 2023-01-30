import catboost
from catboost import CatBoostClassifier
from catboost import CatBoostRegressor


class CatboostModel:
    def __init__(self, parameters: list, model_type: str):
        self.parameters = parameters
        self.model_type = model_type
        if self.model_type == "classification":
            self.model = CatBoostClassifier()
        elif self.model_type == "regression":
            self.model = CatBoostRegressor()

    def fit(self, X, y):
        hyperparams = {param.name: param.value for param in self.parameters}
        self.model.set_params(**hyperparams)
        self.model.fit(X, y)

    def transform(self, X):
        return self.model.predict(X)

    def predict(self, X):
        return self.model.predict(X)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def get_params(self):
        return self.parameters
