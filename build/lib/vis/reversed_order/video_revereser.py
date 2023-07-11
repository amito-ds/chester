import cv2


class VideoReverser:
    def __init__(self, video_data):
        self.video_data = video_data
        self.frames = self.video_data.images
        self.reversed_frames = None
        self.fps = self.video_data.frame_per_second

    def create_reversed_frames(self):
        """
        Reverses the order of the frames in the list.
        """
        reversed_frame_list = self.frames[::-1]
        self.reversed_frames = reversed_frame_list
        return reversed_frame_list

    def run(self, play=False):
        if not self.reversed_frames:
            self.create_reversed_frames()

        height, width, _ = self.reversed_frames[0].shape
        size = (width, height)

        video = cv2.VideoWriter('reversed_output_video.avi', cv2.VideoWriter_fourcc(*'XVID'), self.fps, size)

        for img in self.reversed_frames:
            video.write(img)

        video.release()
        if play:
            try:
                self.play_reversed_video()
            except:
                print("Cannot play video on collab, you can read the video and use the vis module to plot images")
        return video

    def play_reversed_video(self):
        cap = cv2.VideoCapture('reversed_output_video.avi')

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            cv2.imshow('Reversed Video', frame)

            if cv2.waitKey(int(1000 / self.fps)) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
