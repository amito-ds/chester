from sklearn.linear_model import LogisticRegression


class LogisticRegressionModel:
    def __init__(self, parameters: list):
        self.parameters = parameters
        self.logreg = LogisticRegression()

    def fit(self, X, y):
        hyperparams = {param.name: param.value for param in self.parameters}
        self.logreg.set_params(**hyperparams)
        self.logreg.fit(X, y)

    def transform(self, X):
        return self.logreg.predict_proba(X)

    def predict(self, X):
        return self.logreg.predict(X)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def get_params(self):
        return self.parameters
