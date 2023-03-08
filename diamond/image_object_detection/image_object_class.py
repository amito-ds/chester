import torch

from diamond.image_caption.utils import load_images_from_numpy
from diamond.user_classes import ImagesData, ImageDescriptionSpec

import os
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from transformers import pipeline
import timm
from PIL import ImageDraw


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
        self.formatted_images = None
        self.format_images()
        self.load_model()

    def load_model(self):
        self.model = get_image_object_detection_model()

    def format_images(self):
        print("=>Formatting images...")
        self.formatted_images = load_images_from_numpy(self.images_data.raw_images)

    def detect_objects(self):
        bounding_boxes = []
        for image in self.formatted_images:
            # TODO: make sure the right format...
            # test on collab first
            bounding_box = self.model(image)
            bounding_boxes.append(bounding_box)
        return bounding_boxes

    def plot_detected_objets(self):
        pass

    def run(self):
        self.diamond_collector["object detection"] = self.detect_objects()
        if self.plot:
            self.plot_detected_objets()


def get_image_object_detection_model():
    model_name = "facebook/detr-resnet-50"
    model_revision = "main"
    model = pipeline("object-detection", model=model_name, revision=model_revision)
    return model
