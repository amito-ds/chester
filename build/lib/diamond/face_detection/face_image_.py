import torch
from PIL import Image
from PIL import ImageDraw
from matplotlib import pyplot as plt

from diamond.user_classes import ImagesData


class ImageFaceDetection:
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
        self.diamond_collector = {} if diamond_collector is None else diamond_collector
        self.images = self.images_data.raw_images
        self.load_model()

    def load_model(self):
        self.model = get_face_detection_model()

    # TODO: GPU support?
    def detect_yolo_faces(self):
        bounding_boxes = []

        for image in self.images:
            if image.mode != "RGB":
                image = image.convert(mode="RGB")

            # Apply YOLO
            results = self.model(image)

            # Extract the bounding boxes of the detected faces
            faces = results.xyxy[0][results.xyxy[0][:, -1] == 0][:, :4].tolist()
            bounding_boxes.append(faces)
        return bounding_boxes

    def plot_detected_objects(self):
        if not self.plot:
            return None

        fig, axes = plt.subplots(nrows=len(self.images_data.images_to_show),
                                 figsize=(15, 11 * len(self.images_data.images_to_show)))

        for i, (image_data, ax) in enumerate(zip(self.images_data.images_to_show, axes)):
            try:
                image = Image.fromarray(image_data)
            except TypeError:
                image = image_data
            bounding_boxes = self.diamond_collector["face detection"][i]
            # we want to see all of them
            if len(bounding_boxes) == 0:
                continue

            labels = []
            draw = ImageDraw.Draw(image)
            for prediction in bounding_boxes:
                box = prediction
                label = "Object"
                labels.append(label)
                draw.rectangle([box[0], box[1], box[0] + box[2], box[1] + box[3]], outline="red", width=2)
                draw.text((box[0], box[1] - 15), label, fill="black")

            ax.imshow(image)
            ax.set_title(f"Objects detected: {len(bounding_boxes)}")
            ax.axis('off')

            if i >= self.plot_sample:
                break
        plt.show()
        plt.close()

    def run(self):
        self.diamond_collector["face detection"] = self.detect_yolo_faces()
        if self.plot:
            self.plot_detected_objects()


def get_face_detection_model():
    # Load the YOLOv5s model for face detection
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    return model
