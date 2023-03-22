import cv2


class VideoToGrayScale:
    def __init__(self, video_data):
        self.video_data = video_data
        self.frames = self.video_data.images
        self.grayscale_frames = None
        self.fps = self.video_data.frame_per_second

    def frames_to_grayscale(self):
        """
        Converts a list of frames to grayscale using OpenCV.
        """
        grayscale_frame_list = []

        # loop through each frame in the list
        for frame in self.frames:
            # convert the frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grayscale_frame_list.append(gray_frame)
        self.grayscale_frames = grayscale_frame_list
        return grayscale_frame_list

    def run(self, play=False):
        if not self.grayscale_frames:
            self.frames_to_grayscale()

        height, width = self.grayscale_frames[0].shape
        size = (width, height)

        video = cv2.VideoWriter('grayscale_output_video.avi', cv2.VideoWriter_fourcc(*'XVID'), self.fps, size,
                                isColor=False)

        for img in self.grayscale_frames:
            video.write(img)

        video.release()

        if play:
            try:
                self.play_grayscale_video()
            except:
                print("Cannot play video on collab, you can read the video and use the vis module to plot images")
        return video

    def play_grayscale_video(self):
        cap = cv2.VideoCapture('grayscale_output_video.avi')

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            cv2.imshow('Grayscale Video', frame)

            if cv2.waitKey(int(1000 / self.fps)) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
