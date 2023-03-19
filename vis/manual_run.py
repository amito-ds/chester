import cv2
import numpy as np
import os

import numpy as np
import pandas as pd
from PIL import Image

from vis.run import run
from vis.user_classes import *

# # define directory path
# data_dir = '/Users/amitosi/PycharmProjects/chester/chester/data/videos/nature.mp4'
data_dir = '/Users/amitosi/PycharmProjects/chester/chester/data/videos/VID-20230316-WA0001.mp4'
# Create a video capture object
cap = cv2.VideoCapture(data_dir)

# Define image size
image_shape = (3, 4096, 4096)

vis_collector = run(video=cap,
                    frame_per_second=5, image_shape=image_shape, plot_sample=16,
                    get_grayscale=True, get_reveresed_video=True,
                    get_zoomed_video=True, zoom_factor=1.5, zoom_center=(500, 500),
                    get_frames_description=False, detect_frame_objects=False, detect_faces=False, plot=True)
