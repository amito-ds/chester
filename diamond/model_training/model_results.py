from diamond.model_training.train_single import ImageModelTraining


class ImageModelResults:
    def __init__(self, model, train_loss, val_loss, network_parameters,
                 image_model_training: ImageModelTraining,
                 evaluate_predictions):
        self.model = model
        self.train_loss = train_loss
        self.val_loss = val_loss
        self.network_parameters = network_parameters
        self.image_model_training = image_model_training
        self.evaluate_predictions = evaluate_predictions
