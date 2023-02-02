import graphviz
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor, plot_tree

from chester.model_training.data_preparation import CVData
from chester.zero_break.problem_specification import DataInfo


class ModelWeaknesses:
    def __init__(self, cv_data: CVData, data_info: DataInfo, model):
        self.cv_data = cv_data
        self.data_info = data_info
        self.model = model
        self.X_train = self.cv_data.train_data.drop(columns=[self.cv_data.target_column])
        self.y_train = self.cv_data.train_data[self.cv_data.target_column]
        self.X_test = self.cv_data.test_data.drop(columns=[self.cv_data.target_column])
        self.y_test = self.cv_data.test_data[self.cv_data.target_column]
        # retrain the model
        print("retraining model...")
        self.model.retrain(self.X_train, self.y_train)
        # predicting
        self.predict_test = self.model.predict(self.X_test)
        self.predict_test = self.handle_predictions()
        self.error = self.calculate_error()

    def handle_predictions(self):
        y = self.predict_test
        if isinstance(y, pd.Series):
            return y
        elif y.ndim == 2 and y.shape[1] == 1:
            return pd.Series(y.flatten()).reset_index(drop=True)
        elif y.ndim == 2 and y.shape[1] == 2:
            return pd.Series(np.argmax(y, axis=1))
        return y

    def calculate_error(self):
        if 'class' in self.data_info.problem_type_val.lower():
            print("examples for calculate error")
            y_test = self.y_test.reset_index(drop=True)
            try:
                predict_test = self.predict_test.reset_index(drop=True)
            except:
                predict_test = self.predict_test
            print(predict_test[0:10])
            print(y_test[0:10])
            error_per_row = 1 - (y_test == predict_test)
            print("error")
            print(error_per_row)
            return pd.Series(error_per_row, name='error')
        elif 'regression' in self.data_info.problem_type_val.lower():
            y_test = self.y_test.reset_index(drop=True)
            try:
                predict_test = self.predict_test.reset_index(drop=True)
            except:
                predict_test = self.predict_test
            smape = 2 * np.abs(y_test - predict_test) / (np.abs(y_test) + np.abs(predict_test))
            smape = smape.replace([np.inf, -np.inf], np.nan)
            smape = smape.dropna()
            return pd.Series(smape, name='error')

    def plot_decision_tree_error_regressor(self, min_samples_leaf=0.2, max_depth=2):
        model = DecisionTreeRegressor(min_samples_leaf=min_samples_leaf, max_depth=max_depth)
        model.fit(self.X_test, self.error)
        plt.figure(figsize=(15, 15))
        plot_tree(model, filled=True, feature_names=self.X_test.columns)
        plt.title('Regression Tree Showing To Detect Segments With High Error\n'
                  '(Light - Low Error, Stronger Orange = Higher Error)')
        plt.show()

    # train a catboost model
