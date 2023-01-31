import catboost
from catboost import CatBoostClassifier

class CatboostModel:
    def __init__(self, parameters: list, model_type: str):
        self.parameters = parameters
        self.model_type = model_type
        print("this is the problem type", self.model_type)
        if "regression" in self.model_type:
            self.model = catboost.CatBoostRegressor()
        elif "classification" in self.model_type:
            self.model = CatBoostClassifier()

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
