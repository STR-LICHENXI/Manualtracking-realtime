import time
from enum import Enum

import numpy as np


class EffectMode(Enum):
    WAITING = "waiting"
    TWO_D = "2d"
    THREE_D = "3d"
    FAN = "fan"


class EffectStateController:

    FINGERTIPS = (
        4,
        8,
        12,
        16,
        20,
    )

    def __init__(
        self,
        long_hold_seconds=0.55,
        touch_enter_ratio=0.72,
        touch_exit_ratio=0.95,
        first_touch_confirm_seconds=0.03,
        visible_spread_ratio=1.45,
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

        self.visible_spread_ratio = float(
            visible_spread_ratio
        )

        self.reset()

    def reset(self):
        self.mode = EffectMode.WAITING
        self.ready_for_first_touch = False
        self.touching = False
        self.touch_started_at = None
        self.hold_started_at = None
        self.has_opened_2d = False
        self.has_opened_3d = False

    @staticmethod
    def _point(hand, index):
        point = hand["landmarks"][index]

        return np.array(
            [
                float(point["px"]),
                float(point["py"]),
            ],
            dtype=np.float32,
        )

    def _palm_width(self, hand):
        point_a = self._point(
            hand,
            5,
        )

        point_b = self._point(
            hand,
            17,
        )

        return float(
            np.linalg.norm(
                point_a - point_b
            )
        )

    def _scale(self, hands):
        widths = [
            self._palm_width(hand)
            for hand in hands
        ]

        widths = [
            width
            for width in widths
            if (
                np.isfinite(width)
                and width > 3
            )
        ]

        if not widths:
            return None

        return float(
            np.mean(widths)
        )

    def _touch_ratio(self, hands):
        if len(hands) != 2:
            return None

        scale = self._scale(
            hands
        )

        if scale is None:
            return None

        first_hand_tips = [
            self._point(
                hands[0],
                index,
            )
            for index in self.FINGERTIPS
        ]

        second_hand_tips = [
            self._point(
                hands[1],
                index,
            )
            for index in self.FINGERTIPS
        ]

        nearest_distance = min(
            float(
                np.linalg.norm(
                    first_point - second_point
                )
            )
            for first_point in first_hand_tips
            for second_point in second_hand_tips
        )

        return (
            nearest_distance
            /
            scale
        )

    def _spread_ratio(self, hands):
        if len(hands) != 2:
            return None

        scale = self._scale(
            hands
        )

        if scale is None:
            return None

        first_center = np.mean(
            [
                self._point(hands[0], 5),
                self._point(hands[0], 9),
                self._point(hands[0], 13),
                self._point(hands[0], 17),
            ],
            axis=0,
        )

        second_center = np.mean(
            [
                self._point(hands[1], 5),
                self._point(hands[1], 9),
                self._point(hands[1], 13),
                self._point(hands[1], 17),
            ],
            axis=0,
        )

        distance = float(
            np.linalg.norm(
                first_center - second_center
            )
        )

        return distance / scale

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

        was_touching = self.touching

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
            and not was_touching
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

    def update(self, hands):
        now = time.perf_counter()

        touching = self._update_touch(
            hands,
            now,
        )

        spread_ratio = self._spread_ratio(
            hands
        )

        visible = (
            not touching
            and spread_ratio is not None
            and spread_ratio
            >=
            self.visible_spread_ratio
        )

        if self.mode == EffectMode.WAITING:
            if (
                len(hands) == 2
                and not touching
            ):
                self.ready_for_first_touch = True

            if (
                self.ready_for_first_touch
                and touching
                and self.touch_started_at is not None
                and (
                    now
                    -
                    self.touch_started_at
                    >=
                    self.first_touch_confirm_seconds
                )
            ):
                self.mode = EffectMode.TWO_D
                self.has_opened_2d = False

        elif self.mode == EffectMode.TWO_D:
            if visible:
                self.has_opened_2d = True

            if (
                self.has_opened_2d
                and self._hold_complete(
                    touching,
                    now,
                )
            ):
                self.mode = EffectMode.THREE_D
                self.has_opened_3d = False
                self.hold_started_at = None

        elif self.mode == EffectMode.THREE_D:
            if visible:
                self.has_opened_3d = True

            if (
                self.has_opened_3d
                and self._hold_complete(
                    touching,
                    now,
                )
            ):
                self.mode = EffectMode.FAN
                self.hold_started_at = None

        return {
            "mode": self.mode,
            "touching": touching,
            "visible": visible,
            "spread_ratio": spread_ratio,
        }