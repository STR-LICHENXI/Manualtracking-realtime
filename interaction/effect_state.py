import time
from enum import Enum

import numpy as np


class EffectMode(Enum):
    WAITING = "waiting"
    TWO_D = "2d"
    THREE_D = "3d"
    FAN = "fan"


class EffectStateController:

    def __init__(
        self,
        long_hold_seconds=1.0,
        touch_enter_ratio=0.45,
        touch_exit_ratio=0.80,
        first_touch_confirm_seconds=0.10,
    ):
        self.long_hold_seconds = float(
            long_hold_seconds
        )

        self.touch_enter_ratio = float(
            touch_enter_ratio
        )

        self.touch_exit_ratio = float(
            touch_exit_ratio
        )

        self.first_touch_confirm_seconds = float(
            first_touch_confirm_seconds
        )

        self.reset()

    def reset(self):
        self.mode = EffectMode.WAITING

        self.ready_for_first_touch = False
        self.touching = False
        self.touch_started_at = None

        self.has_opened_2d = False
        self.has_opened_3d = False

        self.hold_started_at = None

    @staticmethod
    def _point(
        hand,
        index,
    ):
        p = hand["landmarks"][index]

        return np.array(
            [
                p["px"],
                p["py"],
            ],
            dtype=np.float32,
        )

    def _palm_width(
        self,
        hand,
    ):
        a = self._point(
            hand,
            5,
        )

        b = self._point(
            hand,
            17,
        )

        return float(
            np.linalg.norm(
                a - b
            )
        )

    def _touch_ratio(
        self,
        hands,
    ):
        if len(hands) != 2:
            return None

        a = hands[0]
        b = hands[1]

        tips_a = [
            self._point(a, 4),
            self._point(a, 8),
        ]

        tips_b = [
            self._point(b, 4),
            self._point(b, 8),
        ]

        nearest = min(
            float(
                np.linalg.norm(
                    p1 - p2
                )
            )
            for p1 in tips_a
            for p2 in tips_b
        )

        widths = [
            self._palm_width(a),
            self._palm_width(b),
        ]

        widths = [
            w
            for w in widths
            if w > 3
        ]

        if not widths:
            return None

        return (
            nearest
            /
            float(
                np.mean(widths)
            )
        )

    def _update_touch(
        self,
        hands,
        now,
    ):
        ratio = self._touch_ratio(
            hands
        )

        if ratio is None:
            self.touching = False
            self.touch_started_at = None
            self.hold_started_at = None

            return False

        previous = self.touching

        if self.touching:
            self.touching = (
                ratio
                <=
                self.touch_exit_ratio
            )

        else:
            self.touching = (
                ratio
                <=
                self.touch_enter_ratio
            )

        if (
            self.touching
            and
            not previous
        ):
            self.touch_started_at = now

        if not self.touching:
            self.touch_started_at = None

        return self.touching

    def _hold_complete(
        self,
        touching,
        now,
    ):
        if not touching:
            self.hold_started_at = None
            return False

        if self.hold_started_at is None:
            self.hold_started_at = now

        return (
            now
            -
            self.hold_started_at
            >=
            self.long_hold_seconds
        )

    def update(
        self,
        hands,
    ):
        now = time.perf_counter()

        touching = self._update_touch(
            hands,
            now,
        )

        if self.mode == EffectMode.WAITING:

            if (
                len(hands) == 2
                and
                not touching
            ):
                self.ready_for_first_touch = True

            if (
                self.ready_for_first_touch
                and
                touching
                and
                self.touch_started_at is not None
                and
                now
                -
                self.touch_started_at
                >=
                self.first_touch_confirm_seconds
            ):
                self.mode = EffectMode.TWO_D

                print(
                    "[PaperMorphRT] -> 2D"
                )

        elif self.mode == EffectMode.TWO_D:

            if (
                len(hands) == 2
                and
                not touching
            ):
                self.has_opened_2d = True

            if (
                self.has_opened_2d
                and
                self._hold_complete(
                    touching,
                    now,
                )
            ):
                self.mode = EffectMode.THREE_D
                self.has_opened_3d = False
                self.hold_started_at = None

                print(
                    "[PaperMorphRT] -> 3D"
                )

        elif self.mode == EffectMode.THREE_D:

            if (
                len(hands) == 2
                and
                not touching
            ):
                self.has_opened_3d = True

            if (
                self.has_opened_3d
                and
                self._hold_complete(
                    touching,
                    now,
                )
            ):
                self.mode = EffectMode.FAN
                self.hold_started_at = None

                print(
                    "[PaperMorphRT] -> FAN"
                )

        return {
            "mode": self.mode,
            "touching": touching,
        }