from tamtam.ab_info.ab_class import ABInfo
import seaborn as sns
import matplotlib.pyplot as plt


class MetricCorrelation:
    def __init__(self, ab_info: ABInfo):
        # TODO: fix weights
        self.ab_info = ab_info
        self.df = self.ab_info.df  # pandas
        self.side_col = self.ab_info.test_info.get_side_col()[0]  # column string name
        self.control = "A"
        self.treatment = "B"
        self.metrics = self.ab_info.get_metric_cols()  # m1, m2, ... a list of string

    def calculate_correlation(self):
        metrics = self.metrics  # m1, m2, ... a list of string
        cor_mat = self.df[metrics].corr()
        return cor_mat

    def plot_correlation(self):
        cor_mat = self.calculate_correlation()
        sns.heatmap(cor_mat, annot=True, fmt='.2f', cmap='coolwarm')
        plt.title('Metric Correlation Heatmap')
        plt.show()
        plt.close()

    def partial_plots(self):
        df = self.ab_info.df
        metrics = self.ab_info.get_metric_cols()
        lead_metric = self.ab_info.get_lead_metric()
        other_metrics = metrics.copy()
        other_metrics.remove(lead_metric)

        if len(other_metrics) == 1:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.regplot(x=lead_metric, y=other_metrics[0], data=df, scatter_kws={'alpha': 0.5}, ax=ax)
            ax.set_xlabel(lead_metric)
            ax.set_ylabel(other_metrics[0])
            ax.set_title(f'Comparing lead metric: {lead_metric} with: {other_metrics[0]}')
        else:
            n_cols = 2
            n_rows = (len(other_metrics) + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4))
            axes = axes.ravel()

            for i, metric in enumerate(other_metrics):
                sns.regplot(x=lead_metric, y=metric, data=df, scatter_kws={'alpha': 0.5}, ax=axes[i])
                axes[i].set_xlabel(lead_metric)
                axes[i].set_ylabel(metric)
                axes[i].set_title(f'Comparing lead metric: {lead_metric} with: {metric}')

            plt.tight_layout()
            plt.show()

        plt.close()

    def run(self):
        if len(self.metrics) <= 1:
            return None
        print("==========================> Metrics Correlations ")
        self.plot_correlation()
        self.partial_plots()
