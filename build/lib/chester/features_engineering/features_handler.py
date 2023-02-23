import pandas as pd

from chester.features_engineering.feature_handler import FeatureHandler
from chester.run.user_classes import TextFeatureSpec
from chester.zero_break.problem_specification import DataInfo


class FeaturesHandler:
    def __init__(self,
                 data_info: DataInfo,
                 text_feature_extraction: TextFeatureSpec = None):
        self.data_info = data_info
        self.data = data_info.data
        self.target = data_info.target
        self.text_feature_extraction = text_feature_extraction
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
            feature_handler = FeatureHandler(
                column=data[col],
                feature_type=feature_type,
                col_name=col,
                text_feature_extraction=self.text_feature_extraction,
                data_info=self.data_info
            )
            feature_handlers.append(feature_handler)
        return feature_handlers

    def transform(self):
        feature_types = {'numeric': [], 'categorical': []}
        feature_handlers = self._get_features_handler(data=self.data)
        print(f"Handling {len(feature_handlers)} potential raw features")
        feat_values = []
        feat_names = []
        for feature_handler in feature_handlers:
            try:
                values, names = feature_handler.run()
                feat_values.append(values)
                if feature_handler.feature_type is None:
                    pass
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
            except:
                pass
        final_df = pd.DataFrame()
        for value in feat_values:
            if type(value) == pd.DataFrame:
                final_df = pd.concat([final_df, value], axis=1)
            elif type(value) == pd.Series:
                value = value.to_frame().reset_index(drop=True)
                final_df = pd.concat([final_df, value], axis=1)
        return feature_types, final_df
