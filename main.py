import cv2
import numpy as np
from ultralytics import YOLO


cap = cv2.VideoCapture("video/traffic_1080.mp4")

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolo11n.pt")

area1 = [(0, 600), (0, 1080), (850, 1080), (850, 600)]
area2 = [(1920, 600), (1920, 1080), (920, 1080), (920, 600)]

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Convert area1 to a numpy array
        area1_np = np.array(area1, np.int32)
        area2_np = np.array(area2, np.int32)

        # Draw the polygon on the frame
        cv2.polylines(frame, [area1_np], True, (0, 0, 255), 5)
        cv2.polylines(frame, [area2_np], True, (0, 0, 255), 5)

        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, conf=0.5)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLO11 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
