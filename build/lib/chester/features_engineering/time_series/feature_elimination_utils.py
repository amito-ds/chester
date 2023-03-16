class FeatureEliminationUtils:
    def __init__(self, df):
        self.df = df

    def eliminate_single_values_features(self):
        # Find the features with a single value
        single_value_features = []
        for col in self.df.columns:
            if self.df[col].nunique() == 1:
                single_value_features.append(col)

        # Report the features and remove them from the DataFrame
        if single_value_features:
            print(f"Features with a single value: {', '.join(single_value_features)}")
            self.df = self.df.drop(single_value_features, axis=1)

    def eliminate_high_shared_features(self, p=0.7):
        # Find the features with at least p percent of the same values
        high_correlation_features = []
        for col in self.df.columns:
            if self.df[col].value_counts(normalize=True).iloc[0] >= p:
                high_correlation_features.append(col)

        # Report the features and remove them from the DataFrame
        if high_correlation_features:
            print(f"Features with at least {p * 100}% of the same values: {', '.join(high_correlation_features)}")
            self.df = self.df.drop(high_correlation_features, axis=1)

    def eliminate_duplicates_features(self):
        # Find duplicated features
        duplicates = []
        for i, col1 in enumerate(self.df.columns):
            for col2 in self.df.columns[i + 1:]:
                if self.df[col1].equals(self.df[col2]):
                    duplicates.append(col2)

        # Report the duplicates and remove them from the DataFrame
        if duplicates:
            print(f"Duplicated features: {', '.join(duplicates)}")
            self.df = self.df.drop(duplicates, axis=1)

    def run(self):
        self.eliminate_single_values_features()
        self.eliminate_duplicates_features()
        self.eliminate_high_shared_features()
        return self.df
