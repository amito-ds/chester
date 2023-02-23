import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostRegressor
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor, export_text

from chester.feature_stats.utils import create_pretty_table
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
        self.model.retrain(self.X_train, self.y_train)
        # predicting
        self.predict_test = self.model.predict(self.X_test)
        self.predict_test = self.handle_predictions()
        self.error = self.calculate_error()
        from chester.util import ReportCollector, REPORT_PATH
        self.rc = ReportCollector(REPORT_PATH)

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
            y_test = self.y_test.reset_index(drop=True)
            try:
                predict_test = self.predict_test.reset_index(drop=True)
            except:
                predict_test = self.predict_test
            error_per_row = 1 - (y_test == predict_test)
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
        from sklearn import tree
        from matplotlib import pyplot as plt

        # Prepare the numerical features for processing
        numerical_transformer = SimpleImputer(strategy='median')
        # Prepare the categorical features for processing
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        # Use the ColumnTransformer to apply the transformations to the corresponding features
        transformer = ColumnTransformer(
            transformers=[
                ('c', categorical_transformer, self.data_info.feature_types_val["categorical"]),
                ('n', numerical_transformer, self.data_info.feature_types_val["numeric"])
            ])
        # Apply the transformations to the X_test data
        X_test_transformed = transformer.fit_transform(self.X_test)

        # Train the DecisionTreeRegressor model
        model = DecisionTreeRegressor(min_samples_leaf=min_samples_leaf, max_depth=max_depth)
        model.fit(X_test_transformed, self.error)

        # Get the feature names of the numeric features
        numeric_feature_names = self.data_info.feature_types_val["numeric"]
        # Get the feature names of the categorical features
        categorical_transformer.fit(self.X_test[self.data_info.feature_types_val["categorical"]])
        categories = categorical_transformer.categories_
        categorical_feature_names = []
        for i, col in enumerate(self.data_info.feature_types_val["categorical"]):
            for j in range(len(categories[i])):
                categorical_feature_names.append(f"{col}_{categories[i][j]}")
        # Concatenate the feature names of the categorical and numeric features
        feature_names = numeric_feature_names + categorical_feature_names

        plt.figure(figsize=(13, 13))
        tree.plot_tree(model,
                       feature_names=feature_names,
                       class_names=['error'],
                       rounded=True,
                       filled=True,
                       )
        # Get the tree structure as text
        self.rc.save_object(obj=export_text(model, feature_names=feature_names),
                            text="Print of tree to look for potential segments with high error (use with caution)")
        plt.suptitle("Decision Tree Trained on Model Error")

    def plot_catboost_error_regressor(self, iterations=100, depth=3, learning_rate=0.1):
        model = CatBoostRegressor(iterations=iterations, depth=depth, learning_rate=learning_rate)
        model.fit(self.X_test, self.error, verbose=False)
        plt.figure(figsize=(15, 15))
        feature_importances = np.round(model.feature_importances_, decimals=0)
        feature_imp = pd.DataFrame({'Feature': self.X_test.columns, 'Importance': feature_importances})
        feature_imp = feature_imp.sort_values(by='Importance', ascending=False)
        self.rc.save_object(create_pretty_table(feature_imp[0:50]), text="Most important features by catboost:")
        feature_imp = feature_imp[0:30]
        sns.barplot(x=feature_imp['Importance'], y=feature_imp['Feature'])
        plt.title('CatBoost Feature Importance to Detect Segments with High Error')
        plt.show()
        plt.close()

    def run(self):
        print("Training model to predict the error")
        if np.unique(self.error).size == 1:
            print("🎉 No weaknesses found! All errors on the test set are 0.")
            return None
        self.plot_catboost_error_regressor()
        self.plot_decision_tree_error_regressor()



