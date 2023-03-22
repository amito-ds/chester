import math

import cv2
from matplotlib import pyplot as plt


class VideoData:
    def __init__(self, cap, frame_per_second=None, image_shape=None):
        self.video = cap
        self.image_shape = image_shape
        self.frame_per_second = frame_per_second
        self.images = None
        self.extract_images()

    def extract_images(self):
        # Get the total number of frames in the video
        total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

        # If self.frame_per_second is None, extract all frames
        if self.frame_per_second is None:
            self.images = self._extract_all_frames(total_frames)

        # If self.frame_per_second is a value, extract that many frames per second
        else:
            self.images = self._extract_frames_per_second(total_frames)
        print("Extracted total", len(self.images), "frames")

    def _extract_all_frames(self, total_frames):
        images = []
        for i in range(total_frames):
            # Read the next frame
            ret, frame = self.video.read()

            # If the frame is not valid, break out of the loop
            if not ret:
                break

            # Add the frame to the images list
            images.append(frame)

        return images

    def _extract_frames_per_second(self, total_frames):
        images = []
        fps = self.video.get(cv2.CAP_PROP_FPS)
        duration_sec = total_frames / fps
        num_frames = int(self.frame_per_second * duration_sec)
        frames_to_skip = int(fps / self.frame_per_second)

        # Set the frame position to the beginning of the video
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Loop through the frames and extract every nth frame
        for i in range(num_frames):
            # Skip frames as needed
            for j in range(frames_to_skip):
                ret = self.video.grab()
                if not ret:
                    break

            # Read the next frame
            ret, frame = self.video.retrieve()

            # If the frame is not valid, break out of the loop
            if not ret:
                break

            # Add the frame to the images list
            images.append(frame)

        return images

    def validate(self):
        pass

    def plot_video_images(self, plot_sample):
        total_images = len(self.images)
        plot_sample = min(plot_sample, total_images)
        num_rows = math.ceil(math.sqrt(plot_sample))
        num_cols = math.ceil(plot_sample / num_rows)

        sample_indices = set(range(total_images))  # Get a set of all image indices
        sample_indices = sorted(sample_indices,
                                key=lambda i: i % num_cols)  # Sort the indices by the image order in the video
        sample_indices = sample_indices[:plot_sample]  # Keep only the first `plot_sample` indices

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 10))

        for i, ax in enumerate(axs.flatten()):
            if i < plot_sample:
                index = sample_indices[i]
                image = self.images[index]
                ax.imshow(image)
                ax.set_xticks([])
                ax.set_yticks([])
                # ax.set_title("{:.1f}%".format((index + 1) / plot_sample * 100))

        plt.tight_layout()
        plt.show()
