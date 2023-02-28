import pickle

import numpy as np

from diamond.run import run
from diamond.user_classes import ImageModels, ImageModel

with open("/Users/amitosi/PycharmProjects/chester/chester/data/data_batch_1", 'rb') as fo:
    data = pickle.load(fo, encoding='bytes')

# Extract the image and label data from the batch
images = data[b'data'][0:3000]
labels = np.array(data[b'labels'])[0:3000]

image_model_list = [
    ImageModel(network_name="DenseNet121", batch_size=32, num_epochs=2, optimizer_params={'lr': 0.01},
               remove_num_layers_layers=1),
    ImageModel(network_name="EfficientNetB0", batch_size=32 * 32, num_epochs=2, optimizer_params={'lr': 1})]
image_models = ImageModels(image_model_list=image_model_list)
diamond_collector = run(images,
                        labels,
                        is_augment_data=True,
                        image_models=image_models,
                        plot=False)

print(diamond_collector["models"])
