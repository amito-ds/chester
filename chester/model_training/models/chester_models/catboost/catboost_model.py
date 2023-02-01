from catboost import CatBoostClassifier, CatBoostRegressor


class CatboostModel:
    def __init__(self, parameters: list, model_type: str):
        self.parameters = parameters
        self.model_type = model_type
        # print("this is the model type")
        if "regression" in self.model_type.lower():
            self.model = CatBoostRegressor()
        elif "classification" in self.model_type.lower():
            self.model = CatBoostClassifier()

    # def fit_new

    def fit(self, X, y):
        hyperparams = {param.name: param.value for param in self.parameters}
        self.model.set_params(**hyperparams)
        self.model.fit(X, y)

    def retrain(self, X, y):
        self.model.fit(X, y)

    def transform(self, X):
        return self.model.predict(X)

    def predict(self, X):
        return self.model.predict(X)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def get_params(self):
        # return self.model.get_params()
        # return params
        return self.parameters
