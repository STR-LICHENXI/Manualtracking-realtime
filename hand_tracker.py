from pathlib import Path
import time
import cv2
import mediapipe as mp
import numpy as np


class HandTracker:
    def __init__(
        self,
        model_path: str,
        num_hands: int = 2,
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.model_path = str(Path(model_path))

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=self.model_path
            ),
            running_mode=RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.landmarker = HandLandmarker.create_from_options(options)

        self.start_time = time.perf_counter()
        self.last_timestamp_ms = -1

    def _timestamp_ms(self) -> int:
        timestamp = int(
            (time.perf_counter() - self.start_time) * 1000
        )

        if timestamp <= self.last_timestamp_ms:
            timestamp = self.last_timestamp_ms + 1

        self.last_timestamp_ms = timestamp
        return timestamp

    def process(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb),
        )

        result = self.landmarker.detect_for_video(
            mp_image,
            self._timestamp_ms(),
        )

        hands = []

        if not result.hand_landmarks:
            return hands

        h, w = frame_bgr.shape[:2]

        for i, landmarks in enumerate(result.hand_landmarks):

            handedness = "Unknown"
            handedness_score = 0.0

            if result.handedness and i < len(result.handedness):
                categories = result.handedness[i]

                if categories:
                    handedness = categories[0].category_name
                    handedness_score = categories[0].score

            points = []

            for lm in landmarks:
                points.append(
                    {
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                        "px": int(lm.x * w),
                        "py": int(lm.y * h),
                    }
                )

            thumb = points[4]
            index = points[8]

            pinch_x = int((thumb["px"] + index["px"]) / 2)
            pinch_y = int((thumb["py"] + index["py"]) / 2)

            pinch_distance_px = float(
                np.hypot(
                    thumb["px"] - index["px"],
                    thumb["py"] - index["py"],
                )
            )

            hands.append(
                {
                    "label": handedness,
                    "score": handedness_score,
                    "landmarks": points,
                    "pinch": (pinch_x, pinch_y),
                    "pinch_distance_px": pinch_distance_px,
                }
            )

        return hands

    def close(self):
        if self.landmarker:
            self.landmarker.close()
