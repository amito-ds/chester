import pandas as pd

from chester.data_loader.webtext_data import load_data_pirates, load_data_king_arthur
from chester.features_engineering.feature_handler import FeatureHandler
from chester.zero_break.problem_specification import DataInfo


class FeaturesHandler:
    def __init__(self, data_info: DataInfo):
        self.data = data_info.data
        self.target = data_info.target
        self.problem_type_val = data_info.problem_type_val
        self.feature_types_val = data_info.feature_types_val
        self.loss_detector_val = data_info.loss_detector_val
        self.metrics_detector_val = data_info.metrics_detector_val
        self.model_selection_val = data_info.model_selection_val
        self.label_transformation_val = data_info.label_transformation_val

    def _get_features_handler(self, data: pd.DataFrame = None):
        if data is not None:
            data = self.data
        feature_handlers = []
        for col in data.columns:
            feature_type = None
            for key, value in self.feature_types_val.items():
                if col in value:
                    feature_type = key
                    break
                # Prepare an instance of FeatureHandler
            feature_handler = FeatureHandler(column=data[col], feature_type=feature_type, col_name=col)
            feature_handlers.append(feature_handler)
        return feature_handlers

    def transform(self):
        feature_types = {'numeric': [], 'categorical': []}
        feature_handlers = self._get_features_handler(data=self.data)
        feat_values = []
        feat_names = []
        for feature_handler in feature_handlers:
            values, names = feature_handler.handle_feature()
            feat_values.append(values)
            if feature_handler.feature_type == 'numeric':
                feature_types['numeric'].extend(names)
                feat_names.append(names)
            elif feature_handler.feature_type == 'boolean':
                feature_types['numeric'].extend(names)
                feat_names.append(names)
            elif feature_handler.feature_type == 'text':
                feature_types['numeric'].extend(names)
                feat_names.append(names)
            elif feature_handler.feature_type == 'categorical':
                feat_names.append(names)
                feature_types['categorical'].extend(names)
        final_df = pd.DataFrame()
        for value in feat_values:
            if type(value) == pd.DataFrame:
                final_df = pd.concat([final_df, value], axis=1)
            elif type(value) == pd.Series:
                value = value.to_frame().reset_index(drop=True)
                final_df = pd.concat([final_df, value], axis=1)
        return feature_types, final_df


import numpy as np

df1 = load_data_pirates().assign(target='chat_logs')
df2 = load_data_king_arthur().assign(target='pirates')
df = pd.concat([df1, df2])

# Add numerical column
df["number"] = np.random.uniform(0, 1, df.shape[0])

# Add categorical column
df["categ"] = 'aaa'

# Add boolean column
df["booly"] = True

df.drop(columns='text', inplace=True)

# calc data into
data_info = DataInfo(data=df, target='target')
data_info.calculate()
print(data_info)
# extract features
feat_hand = FeaturesHandler(data_info)
feature_types, final_df = feat_hand.transform()
print(feature_types)
print(final_df)