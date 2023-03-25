# Find image
import os

image_path = os.path.join(os.getcwd(), "dog.jpeg")

# Convert to greyscale
# import Image module
from PIL import Image

img = Image.open(image_path)
gray_img = img.convert("L")

# Flip the image
flipped_image = gray_img.transpose(Image.FLIP_LEFT_RIGHT)

# Show the image
flipped_image.show()
