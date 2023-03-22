import re


class FeatureTypeCategorizer:
    def __init__(self, feature_types: dict):
        self.feature_types = feature_types

    def categorize_features(self):
        new_features = {}
        for key, values in self.feature_types.items():
            category = self.categorize(key)
            if category in new_features:
                new_features[category].update(values)
            else:
                new_features[category] = set(values)
        self.feature_types = new_features
        return self.feature_types

    def categorize_numeric(self, input_string: str):
        input_string = input_string.lower()
        numeric_keywords = ['numeric', 'numerics', 'number', 'numbers', 'numerical', 'numericals', 'num']
        if any(re.search(r'\b' + keyword + r'\b', input_string, re.IGNORECASE)
               for keyword in numeric_keywords):
            return 'numeric'
        return None

    def categorize_boolean(self, input_string: str):
        input_string = input_string.lower()
        boolean_keywords = ['boolean', 'bool', 'booleany']
        if any(re.search(r'\b' + keyword + r'\b', input_string, re.IGNORECASE)
               for keyword in boolean_keywords):
            return 'boolean'
        return None

    def categorize_text(self, input_string: str):
        input_string = input_string.lower()
        text_keywords = ['text', 'nlp']
        if any(re.search(r'\b' + keyword + r'\b', input_string, re.IGNORECASE)
               for keyword in text_keywords):
            return 'text'
        return None

    def categorize_categorical(self, input_string: str):
        input_string = input_string.lower()
        categorical_keywords = ['categorical', 'categoric', 'cat', 'categorical', 'ordinal', 'ordinals']
        if any(re.search(r'\b' + keyword + r'\b', input_string, re.IGNORECASE)
               for keyword in categorical_keywords):
            return 'categorical'
        return None

    def categorize_time(self, input_string: str):
        input_string = input_string.lower()
        time_keywords = ['date', 'time', 'datetime', 'date time', 'time date', 'timestamp', 'time stamp',
                         'timestampped']
        if any(re.search(r'\b' + keyword + r'\b', input_string, re.IGNORECASE)
               for keyword in time_keywords):
            return 'time'
        return None

    def categorize(self, input_string: str):
        numeric = self.categorize_numeric(input_string)
        categorical = self.categorize_categorical(input_string)
        boolean = self.categorize_boolean(input_string)
        text = self.categorize_text(input_string)
        time = self.categorize_time(input_string)
        other = 'other'
        return numeric or categorical or boolean or text or time or other
