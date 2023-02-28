class ImageModelResults:
    def __init__(self, model, train_loss, val_loss, network_parameters):
        self.model = model
        self.train_loss = train_loss
        self.val_loss = val_loss
        self.network_parameters = network_parameters
