from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import numpy as np
import os


def get_image_paths(directory):
    image_paths = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            image_path = os.path.join(directory, filename)
            image_paths.append(image_path)
    return image_paths


images_path = "/Users/amitosi/PycharmProjects/chester/chester/data/weather"

image_path_example = "/Users/amitosi/PycharmProjects/chester/chester/data/weather/sunrise33.jpg"
# Convert the PIL image to a numpy array
image_pil = Image.open(image_path_example)
image_np = np.array(image_pil)
print(image_np.shape)
# exec(open("/Users/amitosi/PycharmProjects/chester/diamond/image_caption/example.py").read())
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

model.save_pretrained("./models")
feature_extractor.save_pretrained("./models")
tokenizer.save_pretrained("./models")

# # Load the pre-trained model and related components from the saved directory
# print("loading models... might take a while")
# model = VisionEncoderDecoderModel.from_pretrained("./models")
# feature_extractor = ViTImageProcessor.from_pretrained("./models")
# tokenizer = AutoTokenizer.from_pretrained("./models")
# # print("Done!")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# max_length = 16
# num_beams = 4
# gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
#
#
# def predict_step(image_paths):
#     images = []
#     count = 0
#     for image_path in image_paths:
#         count = count + 1
#         print("wow")
#         if count > 100:
#             break
#         try:
#             i_image = Image.fromarray(np.uint8(image_paths))
#         except:
#             i_image = Image.open(image_path)
#         print(i_image)
#         try:
#             if i_image.mode != "RGB":
#                 i_image = i_image.convert(mode="RGB")
#         except:
#             pass
#         images.append(i_image)
#
#     pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
#     pixel_values = pixel_values.to(device)
#
#     output_ids = model.generate(pixel_values, **gen_kwargs)
#
#     preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
#     preds = [pred.strip() for pred in preds]
#     return preds
#
#
# # predictions = predict_step(get_image_paths(images_path))
# predictions = predict_step([image_np])
