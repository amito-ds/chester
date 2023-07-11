from matplotlib import pyplot as plt


class ImageInfo:
    def __init__(self, image, label=None):
        self.image = image
        self.label = label

    def get_image_dimensions(self):
        print("self.image.shape", self.image.shape)
        height, width, channels = self.image.shape
        return height, width, channels

    def plot_image(self):
        plt.imshow(self.image)
        plt.axis('off')
        plt.title(f"label = {self.label}")
        plt.show()
