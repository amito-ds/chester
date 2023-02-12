import pandas as pd


class BaselineModel:
    def __init__(self,
                 baseline_value=None,
                 mode_baseline=False,
                 median_baseline=False,
                 avg_baseline=False):
        self.current_baseline = baseline_value
        self.mode_baseline = mode_baseline
        self.median_baseline = median_baseline
        self.avg_baseline = avg_baseline

    def fit(self, y):
        if self.current_baseline is None:
            if self.avg_baseline:
                self.current_baseline = y.mean()
            elif self.median_baseline:
                self.current_baseline = y.quantile(0.5)
            else:
                self.current_baseline = y.mode()[0]

    def transform(self, X):
        return pd.Series([self.current_baseline] * len(X))

    def predict(self, X):
        return self.transform(X)

    def fit_transform(self, X, y):
        self.fit(y)
        return self.transform(X)

    def get_params(self):
        return {"model": "Baseline"}
