import numpy as np
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer


def load_from_path(image_path):
    return Image.open(image_path)


def load_images_from_path(images_path):
    return [load_from_path(image_path) for image_path in images_path]


def load_from_numpy(numpy_image):
    # Convert the numpy array to a supported data type
    numpy_image = np.array(numpy_image)
    # Create a PIL Image object from the numpy array
    image = Image.fromarray(numpy_image)
    # try:
    #     image = Image.fromarray(numpy_image, mode='RGBA')
    # except:
    #     image = Image.fromarray(numpy_image, mode='L')

    return image


def load_images_from_numpy(numpy_images):
    return [load_from_numpy(image_np) for image_np in numpy_images]


def predict_caption(images):
    print("loading models... might take a while")
    model = VisionEncoderDecoderModel.from_pretrained("./models")
    feature_extractor = ViTImageProcessor.from_pretrained("./models")
    tokenizer = AutoTokenizer.from_pretrained("./models")
    # print("Done!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    max_length = 16
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    new_images = []
    for image in images:
        if image.mode != "RGB":
            image = image.convert(mode="RGB")
        new_images.append(image)
    pixel_values = feature_extractor(images=new_images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds
