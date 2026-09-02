import cv2
import numpy as np


class FanRenderer:

    FINGER_TIPS = (4, 8, 12, 16, 20)
    FIST_POINTS = (5, 9, 13, 17)

    def __init__(
        self,
        texture_path,
        line_color=(145, 61, 11),
        line_thickness=1,
        alpha=0.68,
    ):
        texture = cv2.imread(
            str(texture_path),
            cv2.IMREAD_COLOR,
        )

        if texture is None:
            raise FileNotFoundError(
                f"Texture not found: {texture_path}"
            )

        h, w = texture.shape[:2]

        if w > 1200:
            scale = 1200 / float(w)

            texture = cv2.resize(
                texture,
                (
                    1200,
                    int(h * scale),
                ),
                interpolation=cv2.INTER_AREA,
            )

        self.texture = texture
        self.h, self.w = texture.shape[:2]

        self.line_color = tuple(
            int(v) for v in line_color
        )

        self.line_thickness = int(
            line_thickness
        )

        self.alpha = float(alpha)

        cuts = np.linspace(
            0,
            self.h,
            6,
            dtype=int,
        )

        self.parts = [
            self.texture[
                cuts[i]:cuts[i + 1],
                :
            ]
            for i in range(5)
        ]

    @staticmethod
    def _point(hand, index):
        p = hand["landmarks"][index]

        return np.array(
            [
                float(p["px"]),
                float(p["py"]),
            ],
            dtype=np.float32,
        )

    def _geometry(self, hands):
        if len(hands) != 2:
            return None

        data = []

        for hand in hands:
            center = np.mean(
                [
                    self._point(hand, i)
                    for i in self.FIST_POINTS
                ],
                axis=0,
            )

            data.append(
                (
                    center,
                    hand,
                )
            )

        data.sort(
            key=lambda item:
            item[0][0]
        )

        left_hand = data[0][1]
        right_hand = data[1][1]

        apex = np.mean(
            [
                self._point(
                    left_hand,
                    i,
                )
                for i in self.FIST_POINTS
            ],
            axis=0,
        ).astype(np.float32)

        fingers = np.array(
            [
                self._point(
                    right_hand,
                    i,
                )
                for i in self.FINGER_TIPS
            ],
            dtype=np.float32,
        )

        if not np.all(np.isfinite(apex)):
            return None

        if not np.all(np.isfinite(fingers)):
            return None

        return apex, fingers

    def _fill_triangle(
        self,
        frame,
        texture,
        dst_tri,
    ):
        dst_tri = np.float32(dst_tri)

        contour = np.round(
            dst_tri
        ).astype(
            np.int32
        ).reshape(
            -1,
            1,
            2,
        )

        area = abs(
            cv2.contourArea(
                contour
            )
        )

        if area < 50:
            return

        frame_h, frame_w = frame.shape[:2]

        x, y, w, h = cv2.boundingRect(
            contour
        )

        x = max(0, x)
        y = max(0, y)

        w = min(
            w,
            frame_w - x,
        )

        h = min(
            h,
            frame_h - y,
        )

        if w < 3 or h < 3:
            return

        if (
            w * h
            >
            frame_w
            * frame_h
            * 0.80
        ):
            return

        local_tri = (
            dst_tri
            -
            np.array(
                [x, y],
                dtype=np.float32,
            )
        )

        tex_h, tex_w = texture.shape[:2]

        if tex_h < 2 or tex_w < 2:
            return

        src_tri = np.float32([
            [
                0,
                tex_h * 0.5,
            ],
            [
                tex_w - 1,
                0,
            ],
            [
                tex_w - 1,
                tex_h - 1,
            ],
        ])

        matrix = cv2.getAffineTransform(
            src_tri,
            local_tri,
        )

        if not np.all(
            np.isfinite(matrix)
        ):
            return

        warped = cv2.warpAffine(
            texture,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        mask = np.zeros(
            (h, w),
            dtype=np.uint8,
        )

        cv2.fillConvexPoly(
            mask,
            np.round(
                local_tri
            ).astype(np.int32),
            255,
            lineType=cv2.LINE_AA,
        )

        roi = frame[
            y:y + h,
            x:x + w,
        ]

        mask_f = (
            mask.astype(np.float32)
            / 255.0
        )

        mask_f *= self.alpha
        mask_f = mask_f[..., None]

        roi[:] = np.clip(
            roi.astype(np.float32)
            * (1.0 - mask_f)
            +
            warped.astype(np.float32)
            * mask_f,
            0,
            255,
        ).astype(np.uint8)

    def render(
        self,
        frame,
        hands,
    ):
        geometry = self._geometry(
            hands
        )

        if geometry is None:
            return frame, None

        apex, fingers = geometry

        for i in range(5):
            j = (i + 1) % 5

            triangle = np.float32([
                apex,
                fingers[i],
                fingers[j],
            ])

            self._fill_triangle(
                frame,
                self.parts[i],
                triangle,
            )

        apex_pt = tuple(
            np.round(
                apex
            ).astype(int)
        )

        finger_points = [
            tuple(
                np.round(
                    finger
                ).astype(int)
            )
            for finger in fingers
        ]

        for point in finger_points:
            cv2.line(
                frame,
                apex_pt,
                point,
                self.line_color,
                self.line_thickness,
                cv2.LINE_AA,
            )

        for i in range(5):
            j = (i + 1) % 5

            cv2.line(
                frame,
                finger_points[i],
                finger_points[j],
                self.line_color,
                self.line_thickness,
                cv2.LINE_AA,
            )

        cv2.circle(
            frame,
            apex_pt,
            3,
            self.line_color,
            -1,
            cv2.LINE_AA,
        )

        for point in finger_points:
            cv2.circle(
                frame,
                point,
                2,
                self.line_color,
                -1,
                cv2.LINE_AA,
            )

        outer = np.float32([
            apex,
            fingers[0],
            fingers[2],
            fingers[4],
        ])

        return frame, outer