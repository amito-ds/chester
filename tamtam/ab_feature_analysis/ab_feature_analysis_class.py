from chester.run.full_run import run_madcat
from chester.run.user_classes import Data, ModelRun
from tamtam.ab_feature_analysis.ab_catboost import ABCatboostPlotter
from tamtam.ab_feature_analysis.ab_partial_plot import ABPartialPlot
from tamtam.ab_feature_analysis.ab_tree_class import ABTreePlot
from tamtam.ab_info.ab_class import ABInfo
from tamtam.user_class.user_class import TestInfo


class ABFeatureAnalysis:
    def __init__(self, ab_info: ABInfo, test_info: TestInfo):
        self.ab_info = ab_info
        self.test_info = test_info
        self.df = self.ab_info.df
        self.feature_cols = self.test_info.feature_cols or []
        # df[metrics] = df[metrics].multiply(df[side_col].map({'A': -1, 'B': 1}), axis=0)

    def plot_tree(self, metric):
        ab_tree = ABTreePlot(ab_info=self.ab_info, test_info=self.test_info, metric=metric)
        ab_tree.run()

    def plot_catboost(self, metric):
        ab_catboost = ABCatboostPlotter(ab_info=self.ab_info, test_info=self.test_info, metric=metric)
        ab_catboost.run()

    def partial_plot(self, metric):
        ab_partial_plot = ABPartialPlot(ab_info=self.ab_info, test_info=self.test_info, metric=metric)
        ab_partial_plot.run()

    def chester_run(self, metric):
        # prepare data
        df = self.df.copy()
        df[metric] = df[metric].multiply(df[self.test_info.side_col].map({'A': -1, 'B': 1}), axis=0)

        df = df[[metric] + self.feature_cols]
        df.rename(columns={metric: 'target'}, inplace=True)
        # run chester
        run_madcat(data_spec=Data(df=df, target_column='target'), model_run=ModelRun(n_models=3),
                   is_feature_stats=False, is_pre_model=True, is_model_weaknesses=False)

    def run_single(self, metric):
        # self.plot_tree(metric=metric)
        # self.plot_catboost(metric=metric)
        # self.partial_plot(metric=metric)
        self.chester_run(metric=metric)

    def run(self):
        metrics = self.ab_info.get_metric_cols()
        for metric in metrics:
            print(f"==========================> Training a model to predict diff in {metric}")
            self.run_single(metric=metric)
