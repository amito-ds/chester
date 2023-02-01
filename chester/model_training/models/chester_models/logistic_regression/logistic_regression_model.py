from sklearn.linear_model import LogisticRegression


class LogisticRegressionModel:
    def __init__(self, parameters: list):
        self.parameters = parameters
        self.model = LogisticRegression()

    def fit(self, X, y):
        hyperparams = {param.name: param.value for param in self.parameters}
        self.model.set_params(**hyperparams)
        self.model.fit(X, y)

    def retrain(self, X, y):
        self.model.fit(X, y)

    def transform(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def get_params(self):
        return self.parameters
