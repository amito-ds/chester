import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression

from tcap.model_training.models.model_input_and_output_classes import ModelResults

class_name_dict = {
    'LinearRegression': 'Linear Model',
    'LGBMRegressor': 'Light GBM',
    'LogisticRegression': 'Logistic Regression',
    'Sequential': 'Sequential Model',
    'RandomForestRegressor': 'Random Forest',
    'XGBRegressor': 'XGBoost',
    'SVR': 'Support Vector Regression',
    'KNeighborsRegressor': 'K-Neighbors Regression',
    'DecisionTreeRegressor': 'Decision Tree',
    'MLPRegressor': 'Multi-layer Perceptron',
    'AdaBoostRegressor': 'AdaBoost',
    'GradientBoostingRegressor': 'Gradient Boosting',
    'ExtraTreesRegressor': 'Extra Trees',
    'BaggingRegressor': 'Bagging',
    'CatBoostRegressor': 'CatBoost',
    'Lasso': 'Lasso',
    'Ridge': 'Ridge',
    'ElasticNet': 'Elastic Net',
    'PassiveAggressiveRegressor': 'Passive Aggressive'
}


def get_model_name(model):
    model_name = class_name_dict.get(model.__class__.__name__, model.__class__.__name__)
    return model_name


def compare_models_by_type_and_parameters(models_list: list[ModelResults]):
    if len(models_list) <= 1:
        return None
    else:
        # Get a list of lists, where each inner list contains the model tuples of models that are of the same type
        same_type_models = group_models_by_type(models_list)
        # Initialize an empty list to store comparison messages
        comparison_messages = []
        for group in same_type_models:
            # Get the comparison message for each group of models with the same type
            if len(group) == 1:
                comparison_message = type(group[0].model).__name__
            else:
                comparison_message = f"{type(group[0].model).__name__} " \
                                     f"models with parameters: {create_comparison_message(group)}"
            comparison_messages.append(comparison_message)
        print("Choosing the best out of the models:")
        for message in comparison_messages:
            print(message)


def group_models_by_type(models_list: list[ModelResults]):
    # break to same types
    model_types = []
    for model_result in models_list:
        model_types.append(type(model_result.model))
    unique_types = set(model_types)
    same_type_models = []
    for model_type in unique_types:
        same_type_models.append(
            [model_result for model_result in models_list if type(model_result.model) == model_type])
    return same_type_models


# models_list = [ModelResults("Logistic Regression", LogisticRegression(), pd.DataFrame(), [], pd.Series()),
#                ModelResults("Random Forest", RandomForestClassifier(), pd.DataFrame(), [], pd.Series()),
#                ModelResults("SVM", SVC(), pd.DataFrame(), [], pd.Series())]
#
# assert group_models_by_type(models_list) == [
#     [(ModelResults("Logistic Regression", LogisticRegression(), pd.DataFrame(), [], pd.Series()))],
#     [(ModelResults("Random Forest", RandomForestClassifier(), pd.DataFrame(), [], pd.Series()))],
#     [(ModelResults("SVM", SVC(), pd.DataFrame(), [], pd.Series()))]]


def create_comparison_message(models_list):
    param_diff = {}
    for model_name, results, model, parameters in models_list:
        for key, value in parameters.items():
            if key not in param_diff:
                param_diff[key] = [str(value)]
            else:
                param_diff[key].append(str(value))
    param_diff_str = ''
    print(param_diff)
    for key, values in param_diff.items():
        if len(set(values)) > 1:
            param_diff_str += f'({key} = {", ".join(set(values))}) '
    return f'{param_diff_str}'


#
models_list = [
    ("Linear Regression", {'accuracy': 0.9}, LinearRegression(), {'fit_intercept': True, 'normalize': False}),
    ("Logistic Regression", {'accuracy': 0.8}, LogisticRegression(), {'penalty': 'l1', 'C': 0.1}),
    ("Logistic Regression", {'accuracy': 0.9}, LogisticRegression(), {'penalty': 'l2', 'C': 0.2}),
    ("Logistic Regression", {'accuracy': 0.95}, LogisticRegression(), {'penalty': 'l2', 'C': 0.1}),
    ("Random Forest", {'accuracy': 0.7}, RandomForestClassifier(), {'n_estimators': 100, 'max_depth': 5}),
    ("LightGBM", {'accuracy': 0.75}, lgb.LGBMClassifier(), {'boosting_type': 'gbdt', 'num_leaves': 31})]
# # compare_models_by_type(models_list)

# models_list = [('Logistic Regression', {'accuracy': 0.8}, LogisticRegression(), {'penalty': 'l2', 'C': 0.1}),
#                ('Logistic Regression', {'accuracy': 0.9}, LogisticRegression(), {'penalty': 'l2', 'C': 0.2}),
#                ('Logistic Regression', {'accuracy': 0.95}, LogisticRegression(), {'penalty': 'l1', 'C': 0.2})]
# compare_models_by_type_and_parameters(models_list)
# compare_models_by_type_and_parameters(models_list)
# print(create_comparison_message(models_list))
