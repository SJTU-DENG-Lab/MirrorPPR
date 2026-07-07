from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .image_ops import order_points


FFHQ_SHIFT_DIRECTIONS = {
    "Up": (0, -1),
    "Down": (0, 1),
    "Left": (-1, 0),
    "Right": (1, 0),
    "TopLeft": (-1, -1),
    "TopRight": (1, -1),
    "BottomLeft": (-1, 1),
    "BottomRight": (1, 1),
}


class FFHQWildFrameAugmentor:
    def __init__(self, wild_info: dict[str, object], wild_size: tuple[int, int] | None = None):
        if "pixel_size" in wild_info:
            self.img_w, self.img_h = [int(v) for v in wild_info["pixel_size"]]  # type: ignore[index]
        elif wild_size is not None:
            self.img_w, self.img_h = wild_size
        else:
            raise KeyError("FFHQ wild metadata must contain pixel_size when wild_size is not provided.")

        self.face_quad = np.asarray(wild_info["face_quad"], dtype=np.float32)
        if "face_rect" in wild_info:
            rect = [float(v) for v in wild_info["face_rect"]]  # type: ignore[index]
        else:
            xs = self.face_quad[:, 0]
            ys = self.face_quad[:, 1]
            rect = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
        self.face_rect_corners = np.asarray(
            [[rect[0], rect[1]], [rect[2], rect[1]], [rect[2], rect[3]], [rect[0], rect[3]]],
            dtype=np.float32,
        )

    @staticmethod
    def _cross_2d(v1: np.ndarray, v2: np.ndarray) -> float:
        return float(v1[0] * v2[1] - v1[1] * v2[0])

    def _get_line_intersection(self, p: np.ndarray, r: np.ndarray, q: np.ndarray, s: np.ndarray) -> tuple[float | None, float | None]:
        r_cross_s = self._cross_2d(r, s)
        if r_cross_s == 0:
            return None, None
        q_minus_p = q - p
        t = self._cross_2d(q_minus_p, s) / r_cross_s
        u = self._cross_2d(q_minus_p, r) / r_cross_s
        return t, u

    def calculate_max_shift(self, direction_vec: tuple[int, int]) -> float:
        norm = np.linalg.norm(direction_vec)
        if norm == 0:
            return 0.0
        d = np.asarray(direction_vec, dtype=np.float32) / norm

        max_dists = []
        for point in self.face_quad:
            if d[0] > 0:
                max_dists.append((self.img_w - point[0]) / d[0])
            elif d[0] < 0:
                max_dists.append((0 - point[0]) / d[0])
            if d[1] > 0:
                max_dists.append((self.img_h - point[1]) / d[1])
            elif d[1] < 0:
                max_dists.append((0 - point[1]) / d[1])
        limit_outer = max(0.0, min(max_dists) if max_dists else 0.0)

        ray_dir = -d
        limit_inner = float("inf")
        quad_edges = []
        for idx in range(4):
            p1 = self.face_quad[idx]
            p2 = self.face_quad[(idx + 1) % 4]
            quad_edges.append((p1, p2 - p1))

        for rect_pt in self.face_rect_corners:
            point_limit = float("inf")
            for q_start, q_vec in quad_edges:
                t, u = self._get_line_intersection(rect_pt, ray_dir, q_start, q_vec)
                if t is not None and u is not None and 0 <= u <= 1 and t > -1e-5:
                    point_limit = min(point_limit, t)
            limit_inner = min(limit_inner, point_limit)

        return float(min(limit_outer, limit_inner))

    def get_shifted_quad(self, direction_vec: tuple[int, int], magnitude_ratio: float = 1.0) -> np.ndarray:
        norm = np.linalg.norm(direction_vec)
        if norm == 0:
            return self.face_quad.copy()
        unit_vec = np.asarray(direction_vec, dtype=np.float32) / norm
        return self.face_quad + unit_vec * self.calculate_max_shift(direction_vec) * magnitude_ratio


def blend_aligned_face_into_wild(
    aligned_face: Image.Image,
    wild_image: Image.Image,
    face_quad: list[list[float]] | np.ndarray,
    erode_kernel: int = 7,
    blur_kernel: int = 0,
) -> Image.Image:
    aligned = cv2.cvtColor(np.asarray(aligned_face.convert("RGB")), cv2.COLOR_RGB2BGR)
    wild = cv2.cvtColor(np.asarray(wild_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = aligned.shape[:2]
    src_tri = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])
    dst = order_points(np.asarray(face_quad, dtype=np.float32))
    dst_tri = np.float32([dst[0], dst[1], dst[3]])
    matrix = cv2.getAffineTransform(src_tri, dst_tri)
    wild_h, wild_w = wild.shape[:2]
    warped = cv2.warpAffine(aligned, matrix, (wild_w, wild_h), borderMode=cv2.BORDER_REFLECT)
    mask = np.full((h, w), 255, dtype=np.uint8)
    warped_mask = cv2.warpAffine(mask, matrix, (wild_w, wild_h), flags=cv2.INTER_LINEAR)
    if erode_kernel > 0:
        kernel = np.ones((erode_kernel, erode_kernel), np.uint8)
        warped_mask = cv2.erode(warped_mask, kernel)
    if blur_kernel > 0:
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        warped_mask = cv2.GaussianBlur(warped_mask, (blur_kernel, blur_kernel), 0)
    alpha = (warped_mask.astype(np.float32) / 255.0)[..., None]
    blended = warped.astype(np.float32) * alpha + wild.astype(np.float32) * (1.0 - alpha)
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))


def crop_aligned_from_quad(wild_image: Image.Image, face_quad: list[list[float]] | np.ndarray, output_size: int = 1024) -> Image.Image:
    image = cv2.cvtColor(np.asarray(wild_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    src = order_points(np.asarray(face_quad, dtype=np.float32))
    dst = np.asarray([[0, 0], [output_size, 0], [output_size, output_size], [0, output_size]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    crop = cv2.warpPerspective(image, matrix, (output_size, output_size))
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


def load_face_quad(metadata_json: str | Path, image_id: str) -> list[list[float]]:
    import json

    with Path(metadata_json).open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    key = str(int(image_id)) if str(image_id).isdigit() else str(image_id)
    if key not in metadata and str(image_id) in metadata:
        key = str(image_id)
    if key not in metadata:
        raise KeyError(f"Image id {image_id!r} was not found in FFHQ metadata.")
    item = metadata[key]
    if "face_quad" in item:
        return item["face_quad"]
    in_the_wild = item.get("in_the_wild") if isinstance(item, dict) else None
    if isinstance(in_the_wild, dict) and "face_quad" in in_the_wild:
        return in_the_wild["face_quad"]
    image = item.get("image") if isinstance(item, dict) else None
    if isinstance(image, dict) and "face_quad" in image:
        return image["face_quad"]
    else:
        raise KeyError(f"FFHQ metadata entry {image_id!r} does not contain face_quad.")


def load_wild_info(metadata_json: str | Path, image_id: str) -> dict[str, object]:
    import json

    with Path(metadata_json).open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    key = str(int(image_id)) if str(image_id).isdigit() else str(image_id)
    if key not in metadata and str(image_id) in metadata:
        key = str(image_id)
    if key not in metadata:
        raise KeyError(f"Image id {image_id!r} was not found in FFHQ metadata.")
    item = metadata[key]
    wild_info = item.get("in_the_wild") if isinstance(item, dict) else None
    if isinstance(wild_info, dict) and "face_quad" in wild_info:
        return wild_info
    if isinstance(item, dict) and "face_quad" in item:
        return item
    raise KeyError(f"FFHQ metadata entry {image_id!r} does not contain in_the_wild.face_quad.")


def blend_file(
    aligned_face_path: str | Path,
    wild_image_path: str | Path,
    face_quad: list[list[float]],
    output_path: str | Path,
) -> None:
    aligned = Image.open(aligned_face_path).convert("RGB")
    wild = Image.open(wild_image_path).convert("RGB")
    output = blend_aligned_face_into_wild(aligned, wild, face_quad)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_path)
