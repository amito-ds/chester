import cv2


class ImageListToVideo:
    def __init__(self, image_list, fps=30, plot=False):
        self.image_list = image_list
        self.fps = fps
        self.plot = plot
        self.video = self.create_video()
        if self.plot:
            self.play_video()

    def create_video(self):
        if not self.image_list:
            raise ValueError("The image list is empty.")

        height, width, _ = self.image_list[0].shape
        size = (width, height)

        video = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'XVID'), self.fps, size)

        for img in self.image_list:
            video.write(img)

        video.release()
        return video

    def play_video(self):
        cap = cv2.VideoCapture('output_video.avi')

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            cv2.imshow('Video', frame)

            if cv2.waitKey(int(1000 / self.fps)) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
