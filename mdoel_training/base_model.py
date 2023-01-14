class BaseModel:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def train(self):
        pass

    def evaluate(self):
        pass

    def save(self, model_path):
        pass

    def plot_results(self):
        pass
