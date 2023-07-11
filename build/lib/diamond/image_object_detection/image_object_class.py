import torch
from PIL import Image
from PIL import ImageDraw
from matplotlib import pyplot as plt
from transformers import pipeline

from diamond.user_classes import ImagesData


class ImageObjectDetection:
    def __init__(self,
                 images_data: ImagesData,
                 plot=True, plot_sample=10,
                 diamond_collector=None):
        self.images_data = images_data
        self.plot = plot
        self.plot_sample = plot_sample
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_dict = self.images_data.label_dict
        self.model = None
        # TODO: later add support for labels (labels VS description?)
        self.diamond_collector = {} if diamond_collector is None else diamond_collector
        self.images = self.images_data.raw_images
        self.load_model()

    def load_model(self):
        self.model = get_image_object_detection_model()

    def detect_objects(self):
        bounding_boxes = []
        for image in self.images:
            # Run object detection on image
            detections = self.model(image)

            # Append detections to list
            bounding_boxes.append(detections)

        return bounding_boxes

    def plot_detected_objets(self):
        if self.plot:
            try:
                for i in range(len(self.images_data.images_to_show)):
                    try:
                        image = Image.fromarray(self.images_data.images_to_show[i])
                    except:
                        image = self.images_data.images_to_show[i]
                    bounding_boxes = self.diamond_collector["object detection"][i]
                    lables = []
                    # Draw bounding boxes and labels on original image
                    draw = ImageDraw.Draw(image)
                    for prediction in bounding_boxes:
                        box = prediction["box"]
                        label = prediction["label"]
                        lables.append(label)
                        draw.rectangle([box["xmin"], box["ymin"], box["xmax"], box["ymax"]], outline="red", width=12)
                        draw.text((box["xmin"], box["ymin"] - 20), label, fill="black")
                    # Display image with bounding boxes
                    fig, ax = plt.subplots(figsize=(15, 11))
                    plt.title(lables)
                    plt.imshow(image)
                    plt.show()
                    plt.close()
                    if i > self.plot_sample:
                        return None
            except Exception as e:
                print(f"Error while running model: {e}")

    def run(self):
        self.diamond_collector["object detection"] = self.detect_objects()
        if self.plot:
            self.plot_detected_objets()


def get_image_object_detection_model():
    model_name = "facebook/detr-resnet-50"
    model_revision = "main"
    model = pipeline("object-detection", model=model_name, revision=model_revision)
    return model
