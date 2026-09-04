import cv2
import numpy as np


class QuadRenderer:
    def __init__(
        self,
        texture_path,
        min_area=1500.0,
        full_area=10000.0,
        min_hand_distance=80.0,
        min_finger_span=18.0,
        max_frame_area_ratio=0.70,
        surface_opacity=0.60,
        mesh_slices=10,
        rail_color=(255, 175, 20),
        rail_outline=(120, 35, 0),
    ):
        texture = cv2.imread(
            str(texture_path),
            cv2.IMREAD_COLOR,
        )

        if texture is None:
            raise FileNotFoundError(
                f"Could not load texture: {texture_path}"
            )

        h, w = texture.shape[:2]

        if w > 1200:
            scale = 1200 / float(w)

            texture = cv2.resize(
                texture,
                (
                    1200,
                    max(1, int(h * scale)),
                ),
                interpolation=cv2.INTER_AREA,
            )

        self.texture = texture
        self.h, self.w = texture.shape[:2]

        self.min_area = float(min_area)
        self.full_area = float(full_area)
        self.min_hand_distance = float(min_hand_distance)
        self.min_finger_span = float(min_finger_span)
        self.max_frame_area_ratio = float(
            max_frame_area_ratio
        )

        self.surface_opacity = float(
            np.clip(
                surface_opacity,
                0.0,
                1.0,
            )
        )

        self.mesh_slices = max(
            2,
            int(mesh_slices),
        )

        self.rail_color = tuple(
            int(value)
            for value in rail_color
        )

        self.rail_outline = tuple(
            int(value)
            for value in rail_outline
        )

    @staticmethod
    def _xy(hand, index):
        point = hand["landmarks"][index]

        return np.array(
            [
                float(point["px"]),
                float(point["py"]),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _z(hand, index):
        return float(
            hand["landmarks"][index].get(
                "z",
                0.0,
            )
        )

    @staticmethod
    def _area(a, b, c):
        ab = b - a
        ac = c - a

        return abs(
            float(
                ab[0] * ac[1]
                -
                ab[1] * ac[0]
            )
        ) * 0.5

    @staticmethod
    def _smoothstep(a, b, value):
        if b <= a:
            return 1.0

        t = float(
            np.clip(
                (value - a) / (b - a),
                0.0,
                1.0,
            )
        )

        return (
            t
            * t
            * (3.0 - 2.0 * t)
        )

    def _geometry(
        self,
        hands,
        frame_shape,
    ):
        if len(hands) != 2:
            return None

        frame_h, frame_w = frame_shape[:2]

        hands = sorted(
            hands,
            key=lambda hand: (
                self._xy(hand, 4)[0]
                +
                self._xy(hand, 8)[0]
            ) * 0.5,
        )

        left_hand = hands[0]
        right_hand = hands[1]

        left_thumb = self._xy(
            left_hand,
            4,
        )

        left_index = self._xy(
            left_hand,
            8,
        )

        right_thumb = self._xy(
            right_hand,
            4,
        )

        right_index = self._xy(
            right_hand,
            8,
        )

        points = np.array(
            [
                left_thumb,
                left_index,
                right_thumb,
                right_index,
            ],
            dtype=np.float32,
        )

        if not np.all(
            np.isfinite(points)
        ):
            return None

        margin_x = frame_w * 0.08
        margin_y = frame_h * 0.08

        for point in points:
            if (
                point[0] < -margin_x
                or point[0] > frame_w + margin_x
                or point[1] < -margin_y
                or point[1] > frame_h + margin_y
            ):
                return None

        if (
            np.linalg.norm(
                left_index - left_thumb
            )
            <
            self.min_finger_span
        ):
            return None

        if (
            np.linalg.norm(
                right_index - right_thumb
            )
            <
            self.min_finger_span
        ):
            return None

        left_center = (
            left_thumb + left_index
        ) * 0.5

        right_center = (
            right_thumb + right_index
        ) * 0.5

        if (
            np.linalg.norm(
                right_center - left_center
            )
            <
            self.min_hand_distance
        ):
            return None

        for point in (
            left_thumb,
            left_index,
            right_thumb,
            right_index,
        ):
            point[0] = np.clip(
                point[0],
                0,
                frame_w - 1,
            )

            point[1] = np.clip(
                point[1],
                0,
                frame_h - 1,
            )

        left_axis = (
            left_index - left_thumb
        )

        right_axis = (
            right_index - right_thumb
        )

        denominator = max(
            float(
                np.linalg.norm(left_axis)
                *
                np.linalg.norm(right_axis)
            ),
            1e-6,
        )

        twist = (
            left_axis[0] * right_axis[1]
            -
            left_axis[1] * right_axis[0]
        ) / denominator

        return {
            "lt": left_thumb,
            "li": left_index,
            "rt": right_thumb,
            "ri": right_index,
            "ltz": self._z(
                left_hand,
                4,
            ),
            "liz": self._z(
                left_hand,
                8,
            ),
            "rtz": self._z(
                right_hand,
                4,
            ),
            "riz": self._z(
                right_hand,
                8,
            ),
            "twist": float(
                np.clip(
                    twist,
                    -1.0,
                    1.0,
                )
            ),
        }

    def build_quad(
        self,
        hands,
        frame_shape,
    ):
        geometry = self._geometry(
            hands,
            frame_shape,
        )

        if geometry is None:
            return None

        return np.float32(
            [
                geometry["lt"],
                geometry["rt"],
                geometry["ri"],
                geometry["li"],
            ]
        )

    def _blend_triangle(
        self,
        frame,
        source_triangle,
        destination_triangle,
        opacity,
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
            * self.max_frame_area_ratio
        ):
            return

        source = self.texture[
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
        )

        alpha = (
            alpha
            * opacity
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

    def _draw_rails(
        self,
        frame,
        geometry,
    ):
        rails = [
            (
                geometry["lt"],
                geometry["rt"],
            ),
            (
                geometry["li"],
                geometry["ri"],
            ),
        ]

        sides = [
            (
                geometry["lt"],
                geometry["li"],
            ),
            (
                geometry["rt"],
                geometry["ri"],
            ),
        ]

        for start, end in rails:
            point_a = tuple(
                np.round(
                    start
                ).astype(int)
            )

            point_b = tuple(
                np.round(
                    end
                ).astype(int)
            )

            cv2.line(
                frame,
                point_a,
                point_b,
                self.rail_outline,
                5,
                cv2.LINE_AA,
            )

            cv2.line(
                frame,
                point_a,
                point_b,
                self.rail_color,
                2,
                cv2.LINE_AA,
            )

        for start, end in sides:
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
                self.rail_color,
                1,
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
                0.0,
                0.0,
            )

        area = (
            self._area(
                geometry["lt"],
                geometry["rt"],
                geometry["ri"],
            )
            +
            self._area(
                geometry["lt"],
                geometry["ri"],
                geometry["li"],
            )
        )

        frame_h, frame_w = frame.shape[:2]

        if (
            area
            >
            frame_w
            * frame_h
            * self.max_frame_area_ratio
        ):
            return (
                frame,
                None,
                0.0,
                0.0,
            )

        opacity = (
            self._smoothstep(
                self.min_area,
                self.full_area,
                area,
            )
            *
            self.surface_opacity
        )

        quad = np.float32(
            [
                geometry["lt"],
                geometry["rt"],
                geometry["ri"],
                geometry["li"],
            ]
        )

        if opacity <= 0.001:
            return (
                frame,
                quad,
                area,
                opacity,
            )

        triangles = []

        for index in range(
            self.mesh_slices
        ):
            t0 = (
                index
                /
                self.mesh_slices
            )

            t1 = (
                index + 1
            ) / self.mesh_slices

            middle_t = (
                t0 + t1
            ) * 0.5

            left_0 = (
                geometry["lt"]
                +
                (
                    geometry["li"]
                    -
                    geometry["lt"]
                )
                *
                t0
            )

            left_1 = (
                geometry["lt"]
                +
                (
                    geometry["li"]
                    -
                    geometry["lt"]
                )
                *
                t1
            )

            right_0 = (
                geometry["rt"]
                +
                (
                    geometry["ri"]
                    -
                    geometry["rt"]
                )
                *
                t0
            )

            right_1 = (
                geometry["rt"]
                +
                (
                    geometry["ri"]
                    -
                    geometry["rt"]
                )
                *
                t1
            )

            left_z_0 = (
                geometry["ltz"]
                +
                (
                    geometry["liz"]
                    -
                    geometry["ltz"]
                )
                *
                t0
            )

            left_z_1 = (
                geometry["ltz"]
                +
                (
                    geometry["liz"]
                    -
                    geometry["ltz"]
                )
                *
                t1
            )

            right_z_0 = (
                geometry["rtz"]
                +
                (
                    geometry["riz"]
                    -
                    geometry["rtz"]
                )
                *
                t0
            )

            right_z_1 = (
                geometry["rtz"]
                +
                (
                    geometry["riz"]
                    -
                    geometry["rtz"]
                )
                *
                t1
            )

            source_y_0 = (
                self.h - 1
            ) * t0

            source_y_1 = (
                self.h - 1
            ) * t1

            source_top_left = np.array(
                [
                    0.0,
                    source_y_0,
                ],
                dtype=np.float32,
            )

            source_top_right = np.array(
                [
                    self.w - 1,
                    source_y_0,
                ],
                dtype=np.float32,
            )

            source_bottom_right = np.array(
                [
                    self.w - 1,
                    source_y_1,
                ],
                dtype=np.float32,
            )

            source_bottom_left = np.array(
                [
                    0.0,
                    source_y_1,
                ],
                dtype=np.float32,
            )

            twist_depth = (
                geometry["twist"]
                *
                (middle_t - 0.5)
                *
                0.35
            )

            triangles.append(
                (
                    (
                        left_z_0
                        +
                        right_z_0
                        +
                        right_z_1
                    )
                    /
                    3.0
                    +
                    twist_depth,
                    np.float32(
                        [
                            source_top_left,
                            source_top_right,
                            source_bottom_right,
                        ]
                    ),
                    np.float32(
                        [
                            left_0,
                            right_0,
                            right_1,
                        ]
                    ),
                )
            )

            triangles.append(
                (
                    (
                        left_z_0
                        +
                        right_z_1
                        +
                        left_z_1
                    )
                    /
                    3.0
                    +
                    twist_depth,
                    np.float32(
                        [
                            source_top_left,
                            source_bottom_right,
                            source_bottom_left,
                        ]
                    ),
                    np.float32(
                        [
                            left_0,
                            right_1,
                            left_1,
                        ]
                    ),
                )
            )

        triangles.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        output = frame.copy()

        for _, source, destination in triangles:
            self._blend_triangle(
                output,
                source,
                destination,
                opacity,
            )

        self._draw_rails(
            output,
            geometry,
        )

        return (
            output,
            quad,
            area,
            opacity,
        )