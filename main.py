import argparse
import time
import cv2
import numpy as np
import supervision as sp
import torch
from ultralytics import YOLO

COLOURS = sp.ColorPalette.DEFAULT


class VideoProcessor:
    def __init__(
        self,
        weights: str,
        video: str,
        output: str = None,
        conf: float = 0.3,
        iou: float = 0.7,
        device: str = "cpu",
    ) -> None:
        self.source_video = video
        self.output_video = output
        self.confidence_threshold = conf
        self.iou_threshold = iou
        self.device = torch.device(device)
        self.model = YOLO(weights)
        self.box_annotator = sp.BoxAnnotator(color=COLOURS)
        self.label_annotator = sp.LabelAnnotator(color=COLOURS)
        self.tracker = sp.ByteTrack()
        self.drawing = False
        self.polygon_list = []
        self.current_polygon = []
        self.zone_flag = 0

    def draw_polygon(self, event, x, y, flags, param):
        if len(self.current_polygon) == 0:
            self.zone_flag = 0
        elif len(self.current_polygon) == 3:
            self.zone_flag = 1

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_polygon.append((x, y))
            if len(self.current_polygon) > 1:
                cv2.line(
                    param,
                    self.current_polygon[-2],
                    self.current_polygon[-1],
                    (0, 255, 0),
                    2,
                )
            if len(self.current_polygon) == 4:
                self.polygon_list.append(self.current_polygon)
                print("Polygon added:", self.current_polygon)
                cv2.line(
                    param,
                    self.current_polygon[-1],
                    self.current_polygon[0],
                    (0, 255, 0),
                    2,
                )
                self.current_polygon = []
                self.drawing = False

        if event == cv2.EVENT_RBUTTONDOWN:
            self.drawing = False
            self.polygon_list = np.array(self.polygon_list)
            print(self.polygon_list)

    def process_video(self) -> None:
        frame_generator = sp.get_video_frames_generator(
            source_path=self.source_video
        )

        cv2.namedWindow("Frame")

        paused = True
        current_frame = None
        for frame in frame_generator:
            if paused:
                if current_frame is None:
                    current_frame = frame.copy()

                while paused:
                    cv2.setMouseCallback(
                        "Frame", self.draw_polygon, current_frame
                    )

                    for polygon in self.polygon_list:
                        for i in range(len(polygon) - 1):
                            cv2.line(
                                current_frame,
                                polygon[i],
                                polygon[i + 1],
                                (0, 255, 0),
                                2,
                            )
                        cv2.line(
                            current_frame,
                            polygon[-1],
                            polygon[0],
                            (0, 255, 0),
                            2,
                        )

                    processed_frame = self.process_frame(frame=current_frame)
                    cv2.imshow("Frame", processed_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("p"):  # Press 'p' to start/pause
                        paused = False
                        break

            processed_frame = self.process_frame(frame=frame)
            cv2.imshow("Frame", processed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("p"):  # Press 'p' to start/pause
                paused = not paused
                current_frame = frame.copy()
            elif key == ord("q"):  # Press 'q' to quit
                break

        cv2.destroyAllWindows()

    def annotate_frame(
        self, frame: np.ndarray, detections: sp.Detections
    ) -> np.ndarray:
        annotated_frame = frame.copy()
        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
        annotated_frame = self.box_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )
        return annotated_frame

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        result = self.model(
            frame,
            verbose=False,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
        )[0]
        detections = sp.Detections.from_ultralytics(result)
        detections = self.tracker.update_with_detections(detections)
        return self.annotate_frame(frame, detections)


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    parser = argparse.ArgumentParser(
        description="YOLO11 tracking on a video file"
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to the weights file",
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to the video file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        help="Path to the output video file",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.3,
        help="Confidence threshold for detections",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="IoU threshold for detections",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=device,
        help="Device to run inference on (cpu, cuda, mps)",
    )
    args = parser.parse_args()

    processor = VideoProcessor(
        args.weights,
        args.video,
        args.output,
        args.conf,
        args.iou,
        args.device,
    )

    processor.process_video()
