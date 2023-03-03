from diamond.model_training.model_results import ImageModelResults
from diamond.model_training.train_single import ImageModelTraining
from diamond.user_classes import ImagesData, ImageModels


class ImageModelsTraining:
    def __init__(self, images_data: ImagesData,
                 image_models: ImageModels):
        self.images_data = images_data
        self.image_models = image_models
        self.image_model_list = image_models.image_model_list

    def run(self):
        print(f"Training {len(self.image_model_list)} Networks")
        image_model_results = []
        for image_model in self.image_model_list:
            image_model_training = ImageModelTraining(images_data=self.images_data, image_model=image_model)
            model, train_loss, val_loss, evaluate_predictions = image_model_training.run()
            image_model_results.append(ImageModelResults(model=model, train_loss=train_loss, val_loss=val_loss,
                                                         network_parameters=image_model.network_parameters,
                                                         image_model_training=image_model_training,
                                                         evaluate_predictions = evaluate_predictions))
        # sort image_model_results by val loss
        sorted_results = sorted(image_model_results, key=lambda x: x.val_loss)
        return sorted_results
