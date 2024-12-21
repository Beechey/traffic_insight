import cv2
import numpy as np
from ultralytics import YOLO
import torch


def run_track_object():
    cap = cv2.VideoCapture("video/traffic_1080.mp4")
    # cap = cv2.VideoCapture("video/traffic_congested_1080.mp4")

    # Load a pretrained YOLO model (recommended for training)
    model = YOLO("yolo11l.pt")

    area1 = [(0, 600), (0, 1080), (850, 1080), (850, 600)]  # left side
    area2 = [(1920, 600), (1920, 1080), (920, 1080), (920, 600)]  # right side

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            for area in [area1, area2]:
                cv2.polylines(
                    frame, [np.array(area, np.int32)], True, (0, 0, 255), 5
                )

            # Run YOLO11 tracking on the frame, persisting tracks between frames
            results = model.track(
                frame, persist=True, conf=0.25, iou=0.7, device="mps"
            )

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


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    run_track_object()
