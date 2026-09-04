import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["GLOG_minloglevel"] = "2"
os.environ["ABSL_MIN_LOG_LEVEL"] = "2"

warnings.filterwarnings("ignore")

from pathlib import Path

from absl import logging as absl_logging

absl_logging.set_verbosity(absl_logging.ERROR)
absl_logging.set_stderrthreshold(absl_logging.ERROR)

import cv2
import numpy as np

from vision.hand_tracker import HandTracker
from paper.quad_renderer import QuadRenderer
from paper.prism_renderer import PrismRenderer
from paper.fan_renderer import FanRenderer
from interaction.effect_state import (
    EffectMode,
    EffectStateController,
)


try:
    cv2.setLogLevel(2)
except (AttributeError, TypeError):
    pass


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"
TEXTURE_DIR = ROOT / "assets" / "textures"


def find_texture(name):
    extensions = (
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
    )

    for extension in extensions:
        path = TEXTURE_DIR / f"{name}{extension}"

        if path.exists():
            return path

    raise FileNotFoundError(
        f"Cannot find {name} in {TEXTURE_DIR}"
    )


TEXTURE_2D_PATH = find_texture("image1")
TEXTURE_3D_PATH = find_texture("image2")
TEXTURE_FAN_PATH = find_texture("image3")


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def draw_debug_hand(frame, hand):
    landmarks = hand["landmarks"]

    for a, b in HAND_CONNECTIONS:
        pa = (
            landmarks[a]["px"],
            landmarks[a]["py"],
        )

        pb = (
            landmarks[b]["px"],
            landmarks[b]["py"],
        )

        cv2.line(
            frame,
            pa,
            pb,
            (180, 180, 180),
            2,
            cv2.LINE_AA,
        )

    for index, point in enumerate(landmarks):
        radius = (
            6
            if index in (4, 8, 12, 16, 20)
            else 3
        )

        cv2.circle(
            frame,
            (
                point["px"],
                point["py"],
            ),
            radius,
            (255, 255, 255),
            -1,
            cv2.LINE_AA,
        )


def draw_debug_shape(frame, shape):
    if shape is None:
        return

    points = np.round(
        shape
    ).astype(
        np.int32
    )

    if len(points) >= 3:
        cv2.polylines(
            frame,
            [points],
            True,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    tracker = HandTracker(
        model_path=str(MODEL_PATH),
        num_hands=2,
    )

    flat_renderer = QuadRenderer(
        texture_path=TEXTURE_2D_PATH,
        min_area=1500,
        full_area=10000,
        min_hand_distance=80,
        min_finger_span=18,
        max_frame_area_ratio=0.70,
    )

    prism_renderer = PrismRenderer(
        texture_path=TEXTURE_3D_PATH,
        min_area=1500,
        min_hand_distance=80,
        min_finger_span=18,
        depth_scale=0.60,
        back_scale=0.82,
    )

    fan_renderer = FanRenderer(
        texture_path=TEXTURE_FAN_PATH,
    )

    state = EffectStateController(
        long_hold_seconds=0.55,
        touch_enter_ratio=0.72,
        touch_exit_ratio=0.95,
        first_touch_confirm_seconds=0.03,
        visible_spread_ratio=1.45,
    )

    camera = cv2.VideoCapture(
        0,
        cv2.CAP_DSHOW,
    )

    if not camera.isOpened():
        camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        tracker.close()

        raise RuntimeError(
            "Could not open camera."
        )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280,
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720,
    )

    show_debug = False

    try:
        while True:
            success, frame = camera.read()

            if not success:
                break

            frame = cv2.flip(
                frame,
                1,
            )

            hands = tracker.process(
                frame
            )

            status = state.update(
                hands
            )

            shape = None

            if status["visible"]:
                if status["mode"] == EffectMode.TWO_D:
                    (
                        frame,
                        shape,
                        _,
                        _,
                    ) = flat_renderer.render(
                        frame,
                        hands,
                    )

                elif status["mode"] == EffectMode.THREE_D:
                    (
                        frame,
                        shape,
                    ) = prism_renderer.render(
                        frame,
                        hands,
                    )

                elif status["mode"] == EffectMode.FAN:
                    (
                        frame,
                        shape,
                    ) = fan_renderer.render(
                        frame,
                        hands,
                    )

            if show_debug:
                for hand in hands:
                    draw_debug_hand(
                        frame,
                        hand,
                    )

                draw_debug_shape(
                    frame,
                    shape,
                )

            cv2.imshow(
                "ManualTracking Realtime",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == 27 or key == ord("q"):
                break

            if key == ord("d"):
                show_debug = not show_debug

            if key == ord("r"):
                state.reset()

            if key == ord("2"):
                state.mode = EffectMode.TWO_D
                state.has_opened_2d = True

            if key == ord("3"):
                state.mode = EffectMode.THREE_D
                state.has_opened_3d = True

            if key == ord("4"):
                state.mode = EffectMode.FAN

    finally:
        camera.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()