import argparse
from typing import Dict, List, Set, Tuple
import cv2
import numpy as np
import supervision as sp
import torch
from ultralytics import YOLO

COLOURS = sp.ColorPalette.DEFAULT

ZONES_IN = [
    np.array([[693, 955], [936, 993], [925, 1063], [732, 1051]]),
    np.array([(1254, 886), (1333, 693), (1419, 726), (1363, 906)]),
    np.array([(1016, 176), (805, 271), (737, 199), (921, 103)]),
]

ZONES_OUT = [
    np.array([[958, 1009], [1139, 971], [1133, 1062], [938, 1068]]),
    np.array([[1353, 671], [1429, 500], [1533, 550], [1445, 721]]),
    np.array([[720, 307], [557, 397], [522, 307], [706, 222]]),
]


def instantiate_polygon_zones(
    polygons: List[np.array],
    triggering_anchors: sp.Position = sp.Position.CENTER,
) -> List[sp.PolygonZone]:
    return [
        sp.PolygonZone(
            polygon=polygon,
            triggering_anchors=[triggering_anchors],  # Pass as a list
        )
        for polygon in polygons
    ]


class DetectionsManager:
    def __init__(self) -> None:
        self.tracker_id_to_zone_id: Dict[int, int] = {}
        self.recorded_paths: Dict[int, Dict[int, Set]] = {}

    def update(
        self,
        detections: sp.Detections,
        detections_zone_in: List[sp.Detections],
        detections_zone_out: List[sp.Detections],
    ) -> sp.Detections:
        for zone_in_id, detections_in in enumerate(detections_zone_in):
            for tracker_id in detections_in.tracker_id:
                self.tracker_id_to_zone_id.setdefault(tracker_id, zone_in_id)

        for zone_out_id, detections_out in enumerate(detections_zone_out):
            for tracker_id in detections_out.tracker_id:
                if tracker_id in self.tracker_id_to_zone_id:
                    zone_in = self.tracker_id_to_zone_id[tracker_id]
                    self.recorded_paths.setdefault(zone_out_id, {})
                    self.recorded_paths[zone_out_id].setdefault(zone_in, set())
                    self.recorded_paths[zone_out_id][zone_in].add(tracker_id)

        detections.class_id = np.vectorize(
            lambda x: self.tracker_id_to_zone_id.get(x, -1)
        )(detections.tracker_id)
        return detections[detections.class_id != -1]


class VideoProcessor:
    def __init__(
        self,
        weights: str,
        video: str,
        output: str,
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
        self.trace_annotator = sp.TraceAnnotator(
            color=COLOURS, thickness=2, trace_length=100
        )
        self.tracker = sp.ByteTrack()
        self.drawing = False
        self.video_info = sp.VideoInfo.from_video_path(video_path=video)
        self.zones_in = instantiate_polygon_zones(
            polygons=ZONES_IN,
            triggering_anchors=sp.Position.CENTER,
        )
        self.zones_out = instantiate_polygon_zones(
            polygons=ZONES_OUT,
            triggering_anchors=sp.Position.CENTER,
        )
        self.detections_manager = DetectionsManager()

    def process_video(self) -> None:
        frame_generator = sp.get_video_frames_generator(
            source_path=self.source_video
        )

        result = cv2.VideoWriter(
            "output.avi",
            cv2.VideoWriter_fourcc(*"MJPG"),
            10,
            self.video_info.resolution_wh,
        )

        paused = True
        for frame in frame_generator:
            if paused:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("p"):  # Press 'p' to start/pause
                    paused = False
                    break

            processed_frame = self.process_frame(frame=frame)
            cv2.imshow("Frame", processed_frame)

            result.write(processed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("p"):  # Press 'p' to start/pause
                paused = not paused
            elif key == ord("q"):  # Press 'q' to quit
                break

        result.release()
        cv2.destroyAllWindows()

    def annotate_frame(
        self, frame: np.ndarray, detections: sp.Detections
    ) -> np.ndarray:
        annotated_frame = frame.copy()
        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]

        for i, (zone_in, zone_out) in enumerate(
            zip(self.zones_in, self.zones_out)
        ):
            annotated_frame = sp.draw_polygon(
                scene=annotated_frame,
                polygon=zone_in.polygon,
                color=COLOURS.colors[i],
            )

            annotated_frame = sp.draw_polygon(
                scene=annotated_frame,
                polygon=zone_out.polygon,
                color=COLOURS.colors[i],
            )

        annotated_frame = self.box_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )
        annotated_frame = self.trace_annotator.annotate(
            scene=annotated_frame, detections=detections
        )

        for zones_out_id, zone_out in enumerate(self.zones_out):
            zone_centre = sp.get_polygon_center(zone_out.polygon)

            if zones_out_id in self.detections_manager.recorded_paths:
                for zone_in_id in self.detections_manager.recorded_paths:
                    paths = self.detections_manager.recorded_paths[zones_out_id]

                    for i, zone_in_id in enumerate(paths):
                        count = len(paths[zone_in_id])
                        text_anchor = sp.Point(
                            x=zone_centre.x + 40 * i, y=zone_centre.y
                        )
                        annotated_frame = sp.draw_text(
                            scene=annotated_frame,
                            text=f"{count}",
                            text_anchor=text_anchor,
                            background_color=COLOURS.colors[zone_in_id],
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

        detections_zones_in = []
        detections_zones_out = []

        for zone_in, zone_out in zip(self.zones_in, self.zones_out):
            detections_zone_in = detections[zone_in.trigger(detections)]
            detections_zones_in.append(detections_zone_in)

            detections_zone_out = detections[zone_out.trigger(detections)]
            detections_zones_out.append(detections_zone_out)

        detections = self.detections_manager.update(
            detections=detections,
            detections_zone_in=detections_zones_in,
            detections_zone_out=detections_zones_out,
        )

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
        help="Set the confidence threshold for detections",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="Set the IoU threshold for detections",
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
