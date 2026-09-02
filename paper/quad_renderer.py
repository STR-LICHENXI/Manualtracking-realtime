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
        border_color=(145, 61, 11),
        border_thickness=1,
    ):
        self.texture = cv2.imread(
            str(texture_path),
            cv2.IMREAD_COLOR,
        )

        if self.texture is None:
            raise FileNotFoundError(
                f"Could not load texture: {texture_path}"
            )

        self.min_area = float(min_area)
        self.full_area = float(full_area)
        self.min_hand_distance = float(min_hand_distance)
        self.min_finger_span = float(min_finger_span)

        self.border_color = tuple(
            int(v) for v in border_color
        )

        self.border_thickness = int(
            border_thickness
        )

        self.texture_h, self.texture_w = (
            self.texture.shape[:2]
        )

        self.src_tl = np.array(
            [0, 0],
            dtype=np.float32,
        )

        self.src_tr = np.array(
            [self.texture_w - 1, 0],
            dtype=np.float32,
        )

        self.src_br = np.array(
            [
                self.texture_w - 1,
                self.texture_h - 1,
            ],
            dtype=np.float32,
        )

        self.src_bl = np.array(
            [0, self.texture_h - 1],
            dtype=np.float32,
        )

    @staticmethod
    def _point(landmark):
        return np.array(
            [
                float(landmark["px"]),
                float(landmark["py"]),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _smoothstep(a, b, x):
        if b <= a:
            return 1.0

        t = np.clip(
            (x - a) / (b - a),
            0.0,
            1.0,
        )

        return float(
            t * t * (3.0 - 2.0 * t)
        )

    @staticmethod
    def _triangle_area(a, b, c):
        return abs(
            float(
                np.cross(
                    b - a,
                    c - a,
                )
            )
        ) * 0.5

    def build_quad(
        self,
        hands,
        frame_shape,
    ):
        if len(hands) != 2:
            return None

        frame_h, frame_w = frame_shape[:2]

        hands = sorted(
            hands,
            key=lambda hand:
            hand["pinch"][0],
        )

        left = hands[0]
        right = hands[1]

        lt = self._point(
            left["landmarks"][4]
        )

        li = self._point(
            left["landmarks"][8]
        )

        rt = self._point(
            right["landmarks"][4]
        )

        ri = self._point(
            right["landmarks"][8]
        )

        points = [lt, rt, ri, li]

        for p in points:
            if not np.all(
                np.isfinite(p)
            ):
                return None

        left_span = float(
            np.linalg.norm(
                lt - li
            )
        )

        right_span = float(
            np.linalg.norm(
                rt - ri
            )
        )

        if (
            left_span
            <
            self.min_finger_span
        ):
            return None

        if (
            right_span
            <
            self.min_finger_span
        ):
            return None

        left_center = (
            lt + li
        ) * 0.5

        right_center = (
            rt + ri
        ) * 0.5

        hand_distance = float(
            np.linalg.norm(
                right_center
                -
                left_center
            )
        )

        if (
            hand_distance
            <
            self.min_hand_distance
        ):
            return None

        quad = np.float32([
            lt,
            rt,
            ri,
            li,
        ])

        quad[:, 0] = np.clip(
            quad[:, 0],
            0,
            frame_w - 1,
        )

        quad[:, 1] = np.clip(
            quad[:, 1],
            0,
            frame_h - 1,
        )

        return quad

    def _warp_triangle(
        self,
        output,
        src_tri,
        dst_tri,
        alpha,
    ):
        dst_tri = np.float32(
            dst_tri
        )

        src_tri = np.float32(
            src_tri
        )

        area = self._triangle_area(
            dst_tri[0],
            dst_tri[1],
            dst_tri[2],
        )

        if area < 20:
            return

        sx, sy, sw, sh = cv2.boundingRect(
            src_tri
        )

        dx, dy, dw, dh = cv2.boundingRect(
            dst_tri
        )

        frame_h, frame_w = (
            output.shape[:2]
        )

        dx = max(0, dx)
        dy = max(0, dy)

        dw = min(
            dw,
            frame_w - dx,
        )

        dh = min(
            dh,
            frame_h - dy,
        )

        if (
            dw < 2
            or
            dh < 2
        ):
            return

        src_crop = self.texture[
            sy:sy + sh,
            sx:sx + sw,
        ]

        if src_crop.size == 0:
            return

        src_local = src_tri - np.array(
            [sx, sy],
            dtype=np.float32,
        )

        dst_local = dst_tri - np.array(
            [dx, dy],
            dtype=np.float32,
        )

        matrix = cv2.getAffineTransform(
            src_local,
            dst_local,
        )

        if not np.all(
            np.isfinite(matrix)
        ):
            return

        warped = cv2.warpAffine(
            src_crop,
            matrix,
            (dw, dh),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        mask = np.zeros(
            (dh, dw),
            dtype=np.uint8,
        )

        cv2.fillConvexPoly(
            mask,
            np.round(
                dst_local
            ).astype(np.int32),
            255,
            lineType=cv2.LINE_AA,
        )

        roi = output[
            dy:dy + dh,
            dx:dx + dw,
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

        roi[:] = np.clip(
            roi.astype(np.float32)
            *
            (1.0 - mask_f)
            +
            warped.astype(np.float32)
            *
            mask_f,
            0,
            255,
        ).astype(np.uint8)

    def render(
        self,
        frame,
        hands,
    ):
        quad = self.build_quad(
            hands,
            frame.shape,
        )

        if quad is None:
            return (
                frame,
                None,
                0.0,
                0.0,
            )

        lt = quad[0]
        rt = quad[1]
        ri = quad[2]
        li = quad[3]

        area_1 = self._triangle_area(
            lt,
            rt,
            ri,
        )

        area_2 = self._triangle_area(
            lt,
            ri,
            li,
        )

        area = (
            area_1 + area_2
        )

        alpha = self._smoothstep(
            self.min_area,
            self.full_area,
            area,
        )

        if alpha <= 0.001:
            return (
                frame,
                quad,
                area,
                alpha,
            )

        output = frame.copy()

        self._warp_triangle(
            output,
            np.float32([
                self.src_tl,
                self.src_tr,
                self.src_br,
            ]),
            np.float32([
                lt,
                rt,
                ri,
            ]),
            alpha,
        )

        self._warp_triangle(
            output,
            np.float32([
                self.src_tl,
                self.src_br,
                self.src_bl,
            ]),
            np.float32([
                lt,
                ri,
                li,
            ]),
            alpha,
        )

        cv2.line(
            output,
            tuple(
                np.round(
                    lt
                ).astype(int)
            ),
            tuple(
                np.round(
                    rt
                ).astype(int)
            ),
            self.border_color,
            self.border_thickness,
            cv2.LINE_AA,
        )

        cv2.line(
            output,
            tuple(
                np.round(
                    li
                ).astype(int)
            ),
            tuple(
                np.round(
                    ri
                ).astype(int)
            ),
            self.border_color,
            self.border_thickness,
            cv2.LINE_AA,
        )

        cv2.line(
            output,
            tuple(
                np.round(
                    lt
                ).astype(int)
            ),
            tuple(
                np.round(
                    li
                ).astype(int)
            ),
            self.border_color,
            self.border_thickness,
            cv2.LINE_AA,
        )

        cv2.line(
            output,
            tuple(
                np.round(
                    rt
                ).astype(int)
            ),
            tuple(
                np.round(
                    ri
                ).astype(int)
            ),
            self.border_color,
            self.border_thickness,
            cv2.LINE_AA,
        )

        return (
            output,
            quad,
            area,
            alpha,
        )