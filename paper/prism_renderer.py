import cv2
import numpy as np


class PrismRenderer:

    FINGER_TIPS = (4, 8, 12, 16, 20)

    def __init__(
        self,
        texture_path,
        min_area=1500,
        min_hand_distance=80,
        min_finger_span=18,
        depth_scale=0.60,
        back_scale=0.82,
        line_color=(145, 61, 11),
        line_thickness=1,
        alpha=0.75,
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

        if w > 960:
            scale = 960 / float(w)

            texture = cv2.resize(
                texture,
                (
                    960,
                    int(h * scale),
                ),
                interpolation=cv2.INTER_AREA,
            )

        self.texture = texture
        self.texture_h, self.texture_w = texture.shape[:2]

        self.min_hand_distance = 45.0
        self.min_face_area = 60.0

        self.line_color = tuple(
            int(v)
            for v in line_color
        )

        self.line_thickness = int(
            line_thickness
        )

        self.alpha = float(alpha)

        cuts = np.linspace(
            0,
            self.texture_h,
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

    @staticmethod
    def _z(hand, index):
        return float(
            hand["landmarks"][index]["z"]
        )

    def _get_geometry(
        self,
        hands,
        frame_shape,
    ):
        if len(hands) != 2:
            return None

        frame_h, frame_w = frame_shape[:2]

        data = []

        for hand in hands:
            points = np.array(
                [
                    self._point(hand, index)
                    for index in self.FINGER_TIPS
                ],
                dtype=np.float32,
            )

            z_values = np.array(
                [
                    self._z(hand, index)
                    for index in self.FINGER_TIPS
                ],
                dtype=np.float32,
            )

            if not np.all(
                np.isfinite(points)
            ):
                return None

            center = np.mean(
                points,
                axis=0,
            )

            data.append(
                (
                    center,
                    points,
                    z_values,
                )
            )

        data.sort(
            key=lambda item:
            item[0][0]
        )

        left_center, left, left_z = data[0]
        right_center, right, right_z = data[1]

        distance = float(
            np.linalg.norm(
                right_center
                -
                left_center
            )
        )

        if distance < self.min_hand_distance:
            return None

        margin_x = frame_w * 0.08
        margin_y = frame_h * 0.08

        for point in np.vstack(
            (
                left,
                right,
            )
        ):
            if (
                point[0] < -margin_x
                or point[0] > frame_w + margin_x
                or point[1] < -margin_y
                or point[1] > frame_h + margin_y
            ):
                return None

        left[:, 0] = np.clip(
            left[:, 0],
            0,
            frame_w - 1,
        )

        left[:, 1] = np.clip(
            left[:, 1],
            0,
            frame_h - 1,
        )

        right[:, 0] = np.clip(
            right[:, 0],
            0,
            frame_w - 1,
        )

        right[:, 1] = np.clip(
            right[:, 1],
            0,
            frame_h - 1,
        )

        return (
            left,
            right,
            left_z,
            right_z,
        )

    def _valid_quad(
        self,
        quad,
    ):
        if not np.all(
            np.isfinite(quad)
        ):
            return False

        contour = np.round(
            quad
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

        if area < self.min_face_area:
            return False

        if not cv2.isContourConvex(
            contour
        ):
            return False

        edges = [
            np.linalg.norm(
                quad[(i + 1) % 4]
                -
                quad[i]
            )
            for i in range(4)
        ]

        return min(edges) >= 3

    def _fill_quad(
        self,
        frame,
        quad,
        texture,
        alpha,
    ):
        if not self._valid_quad(
            quad
        ):
            return frame

        frame_h, frame_w = frame.shape[:2]

        contour = np.round(
            quad
        ).astype(
            np.int32
        )

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
            return frame

        if (
            w * h
            >
            frame_w
            * frame_h
            * 0.70
        ):
            return frame

        local_quad = (
            quad
            -
            np.array(
                [x, y],
                dtype=np.float32,
            )
        ).astype(
            np.float32
        )

        tex_h, tex_w = texture.shape[:2]

        if tex_h < 2 or tex_w < 2:
            return frame

        src = np.float32([
            [0, 0],
            [tex_w - 1, 0],
            [tex_w - 1, tex_h - 1],
            [0, tex_h - 1],
        ])

        matrix = cv2.getPerspectiveTransform(
            src,
            local_quad,
        )

        if not np.all(
            np.isfinite(matrix)
        ):
            return frame

        warped = cv2.warpPerspective(
            texture,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        mask = np.zeros(
            (h, w),
            dtype=np.uint8,
        )

        cv2.fillConvexPoly(
            mask,
            np.round(
                local_quad
            ).astype(
                np.int32
            ),
            255,
            lineType=cv2.LINE_AA,
        )

        roi = frame[
            y:y + h,
            x:x + w,
        ]

        mask_f = (
            mask.astype(
                np.float32
            )
            /
            255.0
        )

        mask_f *= alpha
        mask_f = mask_f[..., None]

        blended = (
            roi.astype(
                np.float32
            )
            *
            (1.0 - mask_f)
            +
            warped.astype(
                np.float32
            )
            *
            mask_f
        )

        roi[:] = np.clip(
            blended,
            0,
            255,
        ).astype(
            np.uint8
        )

        return frame

    def _draw_structure(
        self,
        frame,
        left,
        right,
    ):
        color = self.line_color

        for i in range(5):
            j = (i + 1) % 5

            cv2.line(
                frame,
                tuple(
                    np.round(
                        left[i]
                    ).astype(int)
                ),
                tuple(
                    np.round(
                        left[j]
                    ).astype(int)
                ),
                color,
                self.line_thickness,
                cv2.LINE_AA,
            )

            cv2.line(
                frame,
                tuple(
                    np.round(
                        right[i]
                    ).astype(int)
                ),
                tuple(
                    np.round(
                        right[j]
                    ).astype(int)
                ),
                color,
                self.line_thickness,
                cv2.LINE_AA,
            )

            cv2.line(
                frame,
                tuple(
                    np.round(
                        left[i]
                    ).astype(int)
                ),
                tuple(
                    np.round(
                        right[i]
                    ).astype(int)
                ),
                color,
                self.line_thickness,
                cv2.LINE_AA,
            )

        for point in np.vstack(
            (
                left,
                right,
            )
        ):
            cv2.circle(
                frame,
                tuple(
                    np.round(
                        point
                    ).astype(int)
                ),
                2,
                color,
                -1,
                cv2.LINE_AA,
            )

        return frame

    def render(
        self,
        frame,
        hands,
    ):
        geometry = self._get_geometry(
            hands,
            frame.shape,
        )

        if geometry is None:
            return frame, None

        (
            left,
            right,
            left_z,
            right_z,
        ) = geometry

        faces = []

        for i in range(5):
            j = (i + 1) % 5

            quad = np.float32([
                left[i],
                right[i],
                right[j],
                left[j],
            ])

            depth = float(
                (
                    left_z[i]
                    +
                    right_z[i]
                    +
                    right_z[j]
                    +
                    left_z[j]
                )
                / 4.0
            )

            faces.append(
                (
                    depth,
                    quad,
                    self.parts[i],
                )
            )

        faces.sort(
            key=lambda item:
            item[0],
            reverse=True,
        )

        for _, quad, texture in faces:
            frame = self._fill_quad(
                frame,
                quad,
                texture,
                self.alpha,
            )

        frame = self._draw_structure(
            frame,
            left,
            right,
        )

        outer = np.float32([
            left[0],
            right[0],
            right[2],
            left[2],
        ])

        return frame, outer