import argparse
import numpy as np
import supervision as sp
import torch
from ultralytics import YOLO


class VideoProcessor:
    def __init__(
        self,
        weights_path: str,
        src_video_path: str,
        output_video_path: str = None,
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.7,
        device: str = "cpu",
    ) -> None:
        self.source_video = src_video_path
        self.output_video = output_video_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = torch.device(device)
        self.model = YOLO(weights_path, device=self.device)

    def process_video(self) -> None:
        pass

    def annotate_frame(
        self, frame: np.ndarray, detections: sp.Detections
    ) -> np.ndarray:
        pass

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        pass


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
