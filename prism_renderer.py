import cv2
import numpy as np


class PrismRenderer:
    FINGER_TIPS = (
        4,
        8,
        12,
        16,
        20,
    )

    def __init__(
        self,
        texture_path,
        min_area=1500,
        min_hand_distance=80,
        min_finger_span=18,
        depth_scale=0.60,
        back_scale=0.82,
        alpha=0.55,
        bridge_color=(255, 245, 40),
        loop_color=(255, 90, 220),
        outline_color=(110, 25, 0),
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
                    max(1, int(h * scale)),
                ),
                interpolation=cv2.INTER_AREA,
            )

        self.texture = texture
        self.h, self.w = texture.shape[:2]

        self.min_hand_distance = max(
            45.0,
            float(min_hand_distance) * 0.55,
        )

        self.min_face_area = max(
            40.0,
            float(min_area) * 0.04,
        )

        self.alpha = float(
            np.clip(
                alpha,
                0.0,
                1.0,
            )
        )

        self.bridge_color = tuple(
            int(value)
            for value in bridge_color
        )

        self.loop_color = tuple(
            int(value)
            for value in loop_color
        )

        self.outline_color = tuple(
            int(value)
            for value in outline_color
        )

        cuts = np.linspace(
            0,
            self.h,
            6,
            dtype=int,
        )

        self.parts = [
            self.texture[
                cuts[index]:cuts[index + 1],
                :
            ].copy()
            for index in range(5)
        ]

    @staticmethod
    def _xy(
        hand,
        index,
    ):
        point = hand["landmarks"][index]

        return np.array(
            [
                float(point["px"]),
                float(point["py"]),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _z(
        hand,
        index,
    ):
        return float(
            hand["landmarks"][index].get(
                "z",
                0.0,
            )
        )

    @staticmethod
    def _area(
        a,
        b,
        c,
    ):
        ab = b - a
        ac = c - a

        return abs(
            float(
                ab[0] * ac[1]
                -
                ab[1] * ac[0]
            )
        ) * 0.5

    def _geometry(
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
                    self._xy(
                        hand,
                        index,
                    )
                    for index in self.FINGER_TIPS
                ],
                dtype=np.float32,
            )

            depths = np.array(
                [
                    self._z(
                        hand,
                        index,
                    )
                    for index in self.FINGER_TIPS
                ],
                dtype=np.float32,
            )

            if not np.all(
                np.isfinite(points)
            ):
                return None

            if not np.all(
                np.isfinite(depths)
            ):
                return None

            data.append(
                (
                    np.mean(
                        points,
                        axis=0,
                    ),
                    points,
                    depths,
                )
            )

        data.sort(
            key=lambda item: item[0][0]
        )

        left_center, left, left_z = data[0]
        right_center, right, right_z = data[1]

        if (
            np.linalg.norm(
                right_center - left_center
            )
            <
            self.min_hand_distance
        ):
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

    def _blend_triangle(
        self,
        frame,
        texture,
        source_triangle,
        destination_triangle,
    ):
        source_triangle = np.float32(
            source_triangle
        )

        destination_triangle = np.float32(
            destination_triangle
        )

        if (
            self._area(
                destination_triangle[0],
                destination_triangle[1],
                destination_triangle[2],
            )
            <
            8.0
        ):
            return

        source_x, source_y, source_w, source_h = (
            cv2.boundingRect(
                source_triangle
            )
        )

        destination_x, destination_y, destination_w, destination_h = (
            cv2.boundingRect(
                destination_triangle
            )
        )

        frame_h, frame_w = frame.shape[:2]

        x1 = max(
            0,
            destination_x,
        )

        y1 = max(
            0,
            destination_y,
        )

        x2 = min(
            frame_w,
            destination_x + destination_w,
        )

        y2 = min(
            frame_h,
            destination_y + destination_h,
        )

        width = x2 - x1
        height = y2 - y1

        if width < 2 or height < 2:
            return

        if (
            width * height
            >
            frame_w
            * frame_h
            * 0.70
        ):
            return

        source = texture[
            source_y:source_y + source_h,
            source_x:source_x + source_w,
        ]

        if source.size == 0:
            return

        source_local = (
            source_triangle
            -
            np.array(
                [
                    source_x,
                    source_y,
                ],
                dtype=np.float32,
            )
        )

        destination_local = (
            destination_triangle
            -
            np.array(
                [
                    x1,
                    y1,
                ],
                dtype=np.float32,
            )
        )

        matrix = cv2.getAffineTransform(
            source_local,
            destination_local,
        )

        if not np.all(
            np.isfinite(matrix)
        ):
            return

        warped = cv2.warpAffine(
            source,
            matrix,
            (
                width,
                height,
            ),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        mask = np.zeros(
            (
                height,
                width,
            ),
            dtype=np.uint8,
        )

        cv2.fillConvexPoly(
            mask,
            np.round(
                destination_local
            ).astype(
                np.int32
            ),
            255,
            lineType=cv2.LINE_AA,
        )

        roi = frame[
            y1:y2,
            x1:x2,
        ]

        alpha = (
            mask.astype(
                np.float32
            )
            /
            255.0
            *
            self.alpha
        )[..., None]

        roi[:] = np.clip(
            roi.astype(
                np.float32
            )
            *
            (1.0 - alpha)
            +
            warped.astype(
                np.float32
            )
            *
            alpha,
            0,
            255,
        ).astype(
            np.uint8
        )

    def _fill_face(
        self,
        frame,
        texture,
        quad,
    ):
        texture_h, texture_w = (
            texture.shape[:2]
        )

        if texture_h < 2 or texture_w < 2:
            return

        top_left = np.array(
            [
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )

        top_right = np.array(
            [
                texture_w - 1,
                0.0,
            ],
            dtype=np.float32,
        )

        bottom_right = np.array(
            [
                texture_w - 1,
                texture_h - 1,
            ],
            dtype=np.float32,
        )

        bottom_left = np.array(
            [
                0.0,
                texture_h - 1,
            ],
            dtype=np.float32,
        )

        self._blend_triangle(
            frame,
            texture,
            np.float32(
                [
                    top_left,
                    top_right,
                    bottom_right,
                ]
            ),
            np.float32(
                [
                    quad[0],
                    quad[1],
                    quad[2],
                ]
            ),
        )

        self._blend_triangle(
            frame,
            texture,
            np.float32(
                [
                    top_left,
                    bottom_right,
                    bottom_left,
                ]
            ),
            np.float32(
                [
                    quad[0],
                    quad[2],
                    quad[3],
                ]
            ),
        )

    @staticmethod
    def _line(
        frame,
        start,
        end,
        color,
        thickness,
    ):
        cv2.line(
            frame,
            tuple(
                np.round(
                    start
                ).astype(int)
            ),
            tuple(
                np.round(
                    end
                ).astype(int)
            ),
            color,
            thickness,
            cv2.LINE_AA,
        )

    def _draw_structure(
        self,
        frame,
        left,
        right,
    ):
        loops = []
        bridges = []

        for index in range(5):
            next_index = (
                index + 1
            ) % 5

            loops.append(
                (
                    left[index],
                    left[next_index],
                )
            )

            loops.append(
                (
                    right[index],
                    right[next_index],
                )
            )

            bridges.append(
                (
                    left[index],
                    right[index],
                )
            )

        for start, end in loops + bridges:
            self._line(
                frame,
                start,
                end,
                self.outline_color,
                7,
            )

        for start, end in loops:
            self._line(
                frame,
                start,
                end,
                self.loop_color,
                2,
            )

        for start, end in bridges:
            self._line(
                frame,
                start,
                end,
                self.bridge_color,
                3,
            )

        for point in np.vstack(
            (
                left,
                right,
            )
        ):
            center = tuple(
                np.round(
                    point
                ).astype(int)
            )

            cv2.circle(
                frame,
                center,
                6,
                self.outline_color,
                -1,
                cv2.LINE_AA,
            )

            cv2.circle(
                frame,
                center,
                3,
                self.bridge_color,
                -1,
                cv2.LINE_AA,
            )

    def render(
        self,
        frame,
        hands,
    ):
        geometry = self._geometry(
            hands,
            frame.shape,
        )

        if geometry is None:
            return (
                frame,
                None,
            )

        left, right, left_z, right_z = (
            geometry
        )

        faces = []

        for index in range(5):
            next_index = (
                index + 1
            ) % 5

            quad = np.float32(
                [
                    left[index],
                    right[index],
                    right[next_index],
                    left[next_index],
                ]
            )

            depth = float(
                (
                    left_z[index]
                    +
                    right_z[index]
                    +
                    right_z[next_index]
                    +
                    left_z[next_index]
                )
                /
                4.0
            )

            faces.append(
                (
                    depth,
                    self.parts[index],
                    quad,
                )
            )

        faces.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        for _, texture, quad in faces:
            self._fill_face(
                frame,
                texture,
                quad,
            )

        self._draw_structure(
            frame,
            left,
            right,
        )

        shape = np.float32(
            [
                left[0],
                right[0],
                right[2],
                left[2],
            ]
        )

        return (
            frame,
            shape,
        )