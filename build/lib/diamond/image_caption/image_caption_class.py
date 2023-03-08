import torch

from diamond.image_caption.utils import load_images_from_numpy
from diamond.user_classes import ImagesData, ImageDescriptionSpec

import os
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer


class ImageDescription:
    def __init__(self,
                 images_data: ImagesData,
                 image_description_spec: ImageDescriptionSpec,
                 plot=True,
                 diamond_collector=None):
        self.images_data = images_data
        self.image_description_spec = image_description_spec
        self.plot = plot
        self.gen_kwargs = self.image_description_spec.gen_kwargs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_dict = self.images_data.label_dict
        self.model, self.feature_extractor, self.tokenizer = None, None, None
        # TODO: later add support for labels (labels VS description?)
        self.diamond_collector = {} if diamond_collector is None else diamond_collector
        self.formatted_images = None
        self.format_images()
        self.load_model()

    def load_model(self):
        self.model, self.feature_extractor, self.tokenizer = get_image_captioning_models()

    def format_images(self):
        print("=>Formatting images...")
        self.formatted_images = load_images_from_numpy(self.images_data.raw_images)

    def predict_caption(self):
        new_images = []
        model = self.model.to(self.device)
        feature_extractor = self.feature_extractor
        tokenizer = self.tokenizer
        gen_kwargs = self.gen_kwargs

        for image in self.formatted_images:
            if image.mode != "RGB":
                image = image.convert(mode="RGB")
            new_images.append(image)
        print("Predicting Image Description")
        pixel_values = feature_extractor(images=new_images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        output_ids = model.generate(pixel_values, **gen_kwargs)
        preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        print("Predicting Image Description - Done!")
        return preds

    def run(self):
        self.diamond_collector["image description"] = self.predict_caption()
        if self.plot:
            pass


def get_image_captioning_models():
    models_dir = "./models"

    # Check if models directory exists
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    print("=> Downloading models...")

    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    print("=> Downloading models - Done!")
    return model, feature_extractor, tokenizer
