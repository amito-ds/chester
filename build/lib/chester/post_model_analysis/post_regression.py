import matplotlib.pyplot as plt
import scipy.stats as stats


class VisualizeRegressionResults:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def scatter_plot(self, ax):
        ax.scatter(self.y_true, self.y_pred, color='blue', alpha=0.5)
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predictions')
        ax.set_title('Scatter Plot of True Values vs Predictions')

    def residual_plot(self, ax):
        residuals = self.y_true - self.y_pred
        ax.scatter(self.y_pred, residuals, color='blue', alpha=0.5)
        ax.set_xlabel('Predictions')
        ax.set_ylabel('Residuals')
        ax.set_title('Residual Plot')
        ax.hlines(y=0, xmin=min(self.y_pred), xmax=max(self.y_pred), color='red', linestyle='--')

    def histogram_plot(self, ax):
        residuals = self.y_true - self.y_pred
        ax.hist(residuals, bins=50, color='blue', alpha=0.5)
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Count')
        ax.set_title('Histogram of Residuals')

    def qq_plot(self, ax):
        residuals = self.y_true - self.y_pred
        stats.probplot(residuals, dist='norm', plot=ax)
        ax.set_title('Q-Q Plot of Residuals')

    def all_plots(self):
        fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        self.scatter_plot(axs[0, 0])
        self.residual_plot(axs[0, 1])
        self.histogram_plot(axs[1, 0])
        self.qq_plot(axs[1, 1])
        plt.show()
        plt.close()
