from catboost import CatBoostClassifier, CatBoostRegressor

from chester.zero_break.problem_specification import DataInfo


class CatboostModel:
    def __init__(self, parameters: list, data_info: DataInfo):
        self.parameters = parameters
        self.model_type = data_info.problem_type_val
        if "regression" in self.model_type.lower():
            # check categorical
            self.model = CatBoostRegressor(cat_features=data_info.feature_types_val["categorical"])
        elif "classification" in self.model_type.lower():
            self.model = CatBoostClassifier(cat_features=data_info.feature_types_val["categorical"])

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
        return self.parameters
