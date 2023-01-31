from sklearn.linear_model import ElasticNet


class LinearRegressionModel:
    def __init__(self, parameters: list, model_type: str):
        self.parameters = parameters
        self.model_type = model_type
        if "regression" in self.model_type.lower():
            self.model = ElasticNet()

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
