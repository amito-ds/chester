import cv2
import numpy as np
import os

import numpy as np
import pandas as pd
from PIL import Image

from vis.run import run
from vis.user_classes import *

# # define directory path
# data_dir = '/Users/amitosi/PycharmProjects/chester/chester/data/videos/production ID_3755687.mp4'
# data_dir = '/Users/amitosi/PycharmProjects/chester/chester/data/videos/ocean-65560.mp4'
# data_dir = '/Users/amitosi/PycharmProjects/chester/chester/data/videos/mixkit-curvy-road-on-a-tree-covered-hill-41537-medium.mp4'
data_dir = '/Users/amitosi/PycharmProjects/chester/chester/data/videos/mixkit-people-dancing-at-a-night-club-4344.mp4'
# Create a video capture object
cap = cv2.VideoCapture(data_dir)

# Define image size
image_shape = (3, 1024, 1024)

vis_collector = run(video=cap,
                    frame_per_second=5, image_shape=image_shape, plot_sample=16,
                    get_frames_description=True, detect_frame_objects=True, detect_faces=True, plot=True)
