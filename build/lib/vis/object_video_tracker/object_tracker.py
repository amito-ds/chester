import cv2


class ObjectVideoTracker:
    def __init__(self, images, object_bounding_box, fps=None):
        self.images = images
        self.object_bounding_box = object_bounding_box
        self.fps = fps

    def play(self):
        # Get the dimensions of the first frame
        height, width, channels = self.images[0].shape
        # Create a VideoWriter object to write the output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('video_object_tracker.avi', fourcc, self.fps or len(self.images), (width, height))
        # Process each frame in the input video
        for i, image in enumerate(self.images):
            bboxes = self.object_bounding_box[i]
            for bbox in bboxes:
                label = bbox['label']
                score = bbox['score']
                x1, y1, x2, y2 = bbox['box']['xmin'], bbox['box']['ymin'], bbox['box']['xmax'], bbox['box']['ymax']
                # Draw a rectangle around the detected object in the frame
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Put a label and score on the top-left corner of the bounding box
                text = f"{label} ({score:.2f})"
                cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            # Write the frame with the bounding boxes to the output video
            out.write(image)
            # Show the frame with the bounding boxes
            cv2.imshow('Object Tracker', image)
            if cv2.waitKey(int(1000 / (self.fps or len(self.images)))) & 0xFF == ord('q'):
                break
        # Release the VideoWriter object and close the output file
        out.release()
        cv2.destroyAllWindows()
