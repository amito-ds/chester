import cv2

data_dir = '/Users/amitosi/PycharmProjects/chester/video_object_tracker.avi'

# Path to the input video file

# Create a VideoCapture object to read the input video
cap = cv2.VideoCapture(data_dir)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()

    # If the frame was read successfully
    if ret:
        # Display the frame
        cv2.imshow('Video', frame)

        # Exit if the 'q' key is pressed
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break
    else:
        # Exit if there are no more frames
        break

# Release the VideoCapture object and close the window
cap.release()
cv2.destroyAllWindows()
