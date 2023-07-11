import cv2


class VideoZoomer:
    def __init__(self, video_data, zoom_factor, zoom_center=None):
        self.video_data = video_data
        self.frames = self.video_data.images
        self.fps = self.video_data.frame_per_second
        self.zoom_factor = zoom_factor
        self.zoom_center = zoom_center
        self.zoomed_frames = None

    def create_zoomed_frames(self):
        """
        Zooms the frames by the specified zoom factor, centered on the specified zoom center (if provided).
        """
        zoomed_frame_list = []
        for frame in self.frames:
            height, width, _ = frame.shape
            new_width = int(width / self.zoom_factor)
            new_height = int(height / self.zoom_factor)

            # Calculate the x and y offsets to center the zoomed frame
            if self.zoom_center is None:
                x_offset = (width - new_width) // 2
                y_offset = (height - new_height) // 2
            else:
                x_offset = max(0, self.zoom_center[0] - (new_width // 2))
                y_offset = max(0, self.zoom_center[1] - (new_height // 2))

            cropped_frame = frame[y_offset:y_offset + new_height, x_offset:x_offset + new_width]
            resized_frame = cv2.resize(cropped_frame, (width, height), interpolation=cv2.INTER_LINEAR)
            zoomed_frame_list.append(resized_frame)

        self.zoomed_frames = zoomed_frame_list
        return zoomed_frame_list

    def run(self, play=False):
        if not self.zoomed_frames:
            self.create_zoomed_frames()

        height, width, _ = self.zoomed_frames[0].shape
        size = (width, height)

        video = cv2.VideoWriter('zoomed_output_video.avi', cv2.VideoWriter_fourcc(*'XVID'), self.fps, size)

        for img in self.zoomed_frames:
            video.write(img)

        video.release()
        if play:
            try:
                self.play_zoomed_video()
            except:
                print("Cannot play video on collab, you can read the video and use the vis module to plot images")
        return video

    def play_zoomed_video(self):
        cap = cv2.VideoCapture('zoomed_output_video.avi')

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            cv2.imshow('Zoomed Video', frame)

            if cv2.waitKey(int(1000 / self.fps)) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
