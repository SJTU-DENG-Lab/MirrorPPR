from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mirrorppr.data.io import read_json


TARGET_AREA_15K = 2211840
TARGET_AREA_BODY_EXTRA = 2396160
BLOCK_SIZE = 32
ROTATION_POOL = [10, 15, 20]
FACE_EXPAND_RATIO = 0.2
BODY_EXPAND_RATIO = 0.05
FACE_AR_RANGE = (0.6, 2.0)
BODY_AR_RANGE = (0.25, 4.0)
MAX_DIM = 4480

BODY_EDIT_PARAMS = {
    "body_shape_right_shoulder",
    "body_shape_swan_neck_left",
    "body_shape_swan_neck_right",
    "body_shape_slim_neck_left",
    "body_shape_slim_neck_right",
    "body_shape_thin_shoulders",
    "body_shape_slim_hand",
    "body_shape_slim_leg",
    "body_shape_slim_waist",
}


def _records_from_json(path: str | Path) -> list[dict[str, Any]]:
    data = read_json(path)
    if isinstance(data, dict):
        return list(data.values())
    if isinstance(data, list):
        return data
    raise TypeError(f"Unsupported JSON root type in {path}: {type(data).__name__}")


def _person_key(record: dict[str, Any]) -> tuple[str, str]:
    if record.get("image_id"):
        image_id = str(record["image_id"])
        if ":" in image_id:
            return tuple(image_id.split(":", 1))  # type: ignore[return-value]
    return str(record.get("group")), str(record.get("index_in_group"))


def _param_key(params: Any) -> tuple[str, ...]:
    if isinstance(params, str):
        return (params,)
    return tuple(str(x) for x in params)


def _resolve_path(path: str | Path, data_root: str | Path | None = None) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    root = Path(data_root) if data_root else Path.cwd()
    return root / path


def _operation_id(record: dict[str, Any]) -> str:
    if record.get("operation_id"):
        return str(record["operation_id"])
    return f"{'+'.join(_param_key(record.get('edit_param', [])))}:{int(record.get('edit_value_level'))}"


def _operation_params_and_level(record: dict[str, Any]) -> tuple[set[str], int]:
    if record.get("operation_id"):
        params, level = parse_operation_spec(str(record["operation_id"]))
        return set(params), level
    return set(_param_key(record.get("edit_param", []))), int(record.get("edit_value_level", 0))


def _source_image_path(record: dict[str, Any]) -> str | None:
    return record.get("source_image_path") or record.get("image_path")


def _geometry_source_image_path(record: dict[str, Any]) -> str | None:
    return record.get("source_image_path") or record.get("image_path")


def _crop_bbox(record: dict[str, Any]) -> list[int] | None:
    return record.get("crop_bbox_in_original") or (record.get("mosaic_info") or {}).get("crop_bbox_in_original")


def _source_tile_path(record: dict[str, Any]) -> str | None:
    return record.get("source_tile_path") or record.get("cropped_mosaic_path")


def _edited_tile_path(record: dict[str, Any]) -> str | None:
    return record.get("edited_tile_path") or record.get("cropped_edited_mosaic_image_path")


def _load_person_geometry(single_op_json: str | Path, data_root: str | Path | None, explicit_path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    single_op_path = Path(single_op_json)
    candidates.append(single_op_path.parent / "person_geometry.json")
    for path in candidates:
        resolved = _resolve_path(path, data_root)
        if resolved.exists():
            data = read_json(resolved)
            if isinstance(data, list):
                return {str(item["image_id"]): item for item in data}
            if isinstance(data, dict):
                return {str(key): value for key, value in data.items()}
            raise TypeError(f"Unsupported person geometry root type in {resolved}: {type(data).__name__}")
    return {}


def parse_operation_spec(spec: str) -> tuple[tuple[str, ...], int]:
    if ":" not in spec:
        raise ValueError(f"Operation must be formatted as param1+param2:level, got {spec!r}")
    params, level = spec.split(":", 1)
    return tuple(p.strip() for p in params.split("+") if p.strip()), int(level)


def diff_mask(before: np.ndarray, after: np.ndarray, threshold: int = 3) -> np.ndarray:
    if before.shape != after.shape:
        before = cv2.resize(before, (after.shape[1], after.shape[0]))
    gray_before = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(cv2.absdiff(gray_before, gray_after), threshold, 255, cv2.THRESH_BINARY)
    return mask


def get_valid_resolutions(area: int, block_size: int, ar_range: tuple[float, float]) -> list[tuple[int, int]]:
    valid = []
    min_w = int((area * ar_range[0]) ** 0.5)
    max_w = int((area * ar_range[1]) ** 0.5)
    start_w = (min_w // block_size) * block_size
    end_w = (max_w // block_size + 1) * block_size
    for width in range(start_w, end_w + 1, block_size):
        if width == 0 or area % width != 0:
            continue
        height = area // width
        if height % block_size == 0 and ar_range[0] <= width / height <= ar_range[1]:
            valid.append((width, height))
    return sorted(valid, key=lambda item: item[0] / item[1])


def select_resolutions_with_mandatory(valid_resolutions: list[tuple[int, int]], mandatory_wh: tuple[int, int]) -> list[tuple[int, int]]:
    selected = sorted(set(valid_resolutions), key=lambda item: item[0] / item[1])
    mw, mh = mandatory_wh
    if mandatory_wh not in selected and mw * mh == TARGET_AREA_15K and mw % BLOCK_SIZE == 0 and mh % BLOCK_SIZE == 0:
        selected.append(mandatory_wh)
        selected.sort(key=lambda item: item[0] / item[1])
    return selected


def largest_inscribed_rect(width: int, height: int, angle_deg: int) -> tuple[int, int, int, int]:
    if angle_deg == 0:
        return 0, 0, width, height
    angle_rad = math.radians(angle_deg)
    cos_a = abs(math.cos(angle_rad))
    sin_a = abs(math.sin(angle_rad))
    new_w = int(width * cos_a + height * sin_a)
    new_h = int(width * sin_a + height * cos_a)
    denom = math.cos(2 * angle_rad)
    w_in = (width * cos_a - height * sin_a) / denom
    h_in = (height * cos_a - width * sin_a) / denom
    if w_in < 0 or h_in < 0:
        w_in, h_in = width * 0.5, height * 0.5
    w_in, h_in = int(w_in), int(h_in)
    return (new_w - w_in) // 2, (new_h - h_in) // 2, w_in, h_in


def rotation_matrix_and_size(width: int, height: int, angle: int) -> tuple[np.ndarray, tuple[int, int]]:
    center_x, center_y = width // 2, height // 2
    matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    new_w = int((height * abs(matrix[0, 1])) + (width * abs(matrix[0, 0])))
    new_h = int((height * abs(matrix[0, 0])) + (width * abs(matrix[0, 1])))
    matrix[0, 2] += (new_w / 2) - center_x
    matrix[1, 2] += (new_h / 2) - center_y
    return matrix, (new_w, new_h)


def transform_points_affine(points: list[list[float]], matrix: np.ndarray) -> list[list[float]]:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.transform(pts, matrix).reshape(-1, 2).tolist()


def farthest_crop_position(
    valid_x_range: tuple[float, float],
    valid_y_range: tuple[float, float],
    crop_wh: tuple[float, float],
    ref_center_xy: tuple[float, float],
) -> tuple[float, float]:
    v_xmin, v_xmax = valid_x_range
    v_ymin, v_ymax = valid_y_range
    crop_w, crop_h = crop_wh
    ref_x, ref_y = ref_center_xy
    candidates = [(v_xmin, v_ymin), (v_xmax, v_ymin), (v_xmin, v_ymax), (v_xmax, v_ymax)]
    return max(candidates, key=lambda point: (point[0] + crop_w / 2 - ref_x) ** 2 + (point[1] + crop_h / 2 - ref_y) ** 2)


def precompute_rotation_data(
    record: dict[str, Any],
    angles: list[int],
    orig_w: int,
    orig_h: int,
    bbox_key: str,
    expand_ratio: float,
) -> dict[int, dict[str, Any]]:
    crop_vertices = record["encoder_image_process_info"]["crop_vertices_on_origin"]
    xs = [pt[0] for pt in crop_vertices]
    ys = [pt[1] for pt in crop_vertices]
    old_center = ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)

    fx, fy, fw, fh = [float(v) for v in record[bbox_key]]
    dw = fw * expand_ratio / 2.0
    dh = fh * expand_ratio / 2.0
    fx1 = max(0, int(fx - dw))
    fy1 = max(0, int(fy - dh))
    fx2 = min(orig_w, int(fx + fw + dw))
    fy2 = min(orig_h, int(fy + fh + dh))
    bbox_corners = [[fx1, fy1], [fx2, fy1], [fx2, fy2], [fx1, fy2]]

    cache = {}
    for angle in set([0] + angles):
        matrix, _ = rotation_matrix_and_size(orig_w, orig_h, angle)
        safe_zone = largest_inscribed_rect(orig_w, orig_h, angle)
        sx, sy, sw, sh = safe_zone
        rot_bbox_corners = transform_points_affine(bbox_corners, matrix)
        rx = [pt[0] for pt in rot_bbox_corners]
        ry = [pt[1] for pt in rot_bbox_corners]
        cache[angle] = {
            "safe_zone": safe_zone,
            "safe_center": (sx + sw / 2.0, sy + sh / 2.0),
            "rot_bbox_rect": [min(rx), min(ry), max(rx), max(ry)],
            "ref_for_diff": old_center if angle == 0 else (sx + sw / 2.0, sy + sh / 2.0),
        }
    return cache


def process_single_professional_augmentation(target_w: int, target_h: int, angle: int, rot_info: dict[str, Any]) -> dict[str, Any] | None:
    target_ar = target_w / target_h
    sx, sy, sw, sh = rot_info["safe_zone"]
    rfx1, rfy1, rfx2, rfy2 = rot_info["rot_bbox_rect"]
    crop_w = sw
    crop_h = crop_w / target_ar
    if crop_h > sh:
        crop_h = sh
        crop_w = crop_h * target_ar

    range_x_min, range_x_max = sx, max(sx, sx + sw - crop_w)
    range_y_min, range_y_max = sy, max(sy, sy + sh - crop_h)
    bbox_x_min, bbox_x_max = max(0, rfx2 - crop_w), rfx1
    bbox_y_min, bbox_y_max = max(0, rfy2 - crop_h), rfy1
    valid_x_min = max(range_x_min, bbox_x_min)
    valid_x_max = min(range_x_max, bbox_x_max)
    valid_y_min = max(range_y_min, bbox_y_min)
    valid_y_max = min(range_y_max, bbox_y_max)
    if valid_x_min > valid_x_max or valid_y_min > valid_y_max:
        return None

    best_x, best_y = farthest_crop_position(
        (valid_x_min, valid_x_max),
        (valid_y_min, valid_y_max),
        (crop_w, crop_h),
        rot_info["ref_for_diff"],
    )
    x1 = int(round(best_x))
    y1 = int(round(best_y))
    x2 = int(round(best_x + crop_w))
    y2 = int(round(best_y + crop_h))
    return {
        "target_resolution": [int(target_w), int(target_h)],
        "crop_vertices_on_origin": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        "scale": target_w / crop_w,
        "rotation": int(angle),
    }


def generate_professional_augmentation_specs(
    record: dict[str, Any],
    image_size: tuple[int, int],
    is_body_edit: bool,
    seed: int,
) -> list[dict[str, Any]]:
    bbox_key = "body_bbox_xywh" if is_body_edit else "face_bbox_xywh"
    if "encoder_image_process_info" not in record or bbox_key not in record:
        raise KeyError(f"Professional augmentation requires encoder_image_process_info and {bbox_key} in the metadata record.")

    orig_w, orig_h = image_size
    orig_target_wh = tuple(record["encoder_image_process_info"]["target_resolution"])
    if is_body_edit:
        resolution_pool = get_valid_resolutions(TARGET_AREA_15K, BLOCK_SIZE, BODY_AR_RANGE)
        resolution_pool += get_valid_resolutions(TARGET_AREA_BODY_EXTRA, BLOCK_SIZE, BODY_AR_RANGE)
        expand_ratio = BODY_EXPAND_RATIO
    else:
        resolution_pool = get_valid_resolutions(TARGET_AREA_15K, BLOCK_SIZE, FACE_AR_RANGE)
        expand_ratio = FACE_EXPAND_RATIO
    selected_resolutions = select_resolutions_with_mandatory(resolution_pool, orig_target_wh)
    rot_cache = precompute_rotation_data(record, ROTATION_POOL, orig_w, orig_h, bbox_key, expand_ratio)
    rng = random.Random(seed)
    augmentations = []
    orig_ar = orig_target_wh[0] / orig_target_wh[1]

    for target_w, target_h in selected_resolutions:
        target_ar = target_w / target_h
        if abs(target_ar - orig_ar) > 0.01:
            spec = process_single_professional_augmentation(target_w, target_h, 0, rot_cache[0])
            if spec:
                augmentations.append(spec)
        rot_angle = rng.choice(ROTATION_POOL)
        sx, sy, sw, sh = rot_cache[rot_angle]["safe_zone"]
        if sh == 0:
            continue
        if abs(target_ar - (sw / sh)) > 0.001:
            spec = process_single_professional_augmentation(target_w, target_h, rot_angle, rot_cache[rot_angle])
            if spec:
                augmentations.append(spec)
    return [aug for aug in augmentations if aug["target_resolution"][0] <= MAX_DIM and aug["target_resolution"][1] <= MAX_DIM]


def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    if angle == 0:
        return image
    height, width = image.shape[:2]
    matrix, (new_w, new_h) = rotation_matrix_and_size(width, height, angle)
    return cv2.warpAffine(image, matrix, (new_w, new_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def warp_professional_image(image: np.ndarray, spec: dict[str, Any], do_flip: bool) -> np.ndarray:
    target_w, target_h = [int(v) for v in spec["target_resolution"]]
    src_pts = np.asarray(spec["crop_vertices_on_origin"], dtype=np.float32)
    dst_pts = np.asarray([[0, 0], [target_w, 0], [target_w, target_h], [0, target_h]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, matrix, (target_w, target_h), flags=cv2.INTER_LANCZOS4)
    if do_flip:
        warped = cv2.flip(warped, 1)
    return warped


def save_warped_pair(
    source: np.ndarray,
    target: np.ndarray,
    spec: dict[str, Any],
    output_dir: str | Path,
    prefix: str,
    rotate: bool = False,
    do_flip: bool = False,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    angle = int(spec.get("rotation", 0)) if rotate else 0
    source_base = rotate_image(source, angle)
    target_base = rotate_image(target, angle)
    source_out_image = warp_professional_image(source_base, spec, do_flip)
    target_out_image = warp_professional_image(target_base, spec, do_flip)
    source_out = output_dir / f"{prefix}_source.png"
    target_out = output_dir / f"{prefix}_target.png"
    cv2.imwrite(str(source_out), source_out_image)
    cv2.imwrite(str(target_out), target_out_image)
    return {"source_image": str(source_out), "target_image": str(target_out)}


def _suppress_tile_border(mask: np.ndarray, record: dict[str, Any]) -> np.ndarray:
    params, level = _operation_params_and_level(record)
    top, bottom, left, right = 2, 2, 2, 2
    if ("face_trans" in params and level == 100) or ("jaw_trans" in params and level == -100) or ({"mandible_left", "mandible_right"} & params and level == -100):
        bottom = 20
    elif ("face_trans" in params and level == -100) or ("jaw_trans" in params and level == 100) or ({"mandible_left", "mandible_right"} & params and level == 100):
        bottom = 5
    height, width = mask.shape[:2]
    if height > top + bottom and width > left + right:
        if top:
            mask[:top, :] = 0
        if bottom:
            mask[-bottom:, :] = 0
        if left:
            mask[:, :left] = 0
        if right:
            mask[:, -right:] = 0
    return mask


def compose_records(
    original_image_path: str | Path,
    records: list[dict[str, Any]],
    output_path: str | Path,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    original_image_path = _resolve_path(original_image_path, data_root)
    original = cv2.imread(str(original_image_path))
    if original is None:
        raise FileNotFoundError(f"Could not read original image: {original_image_path}")
    height, width = original.shape[:2]
    sum_buffer = np.zeros((height, width, 3), dtype=np.float32)
    weight_buffer = np.zeros((height, width, 1), dtype=np.float32)
    used_records = []

    for record in records:
        before_path = _source_tile_path(record)
        after_path = _edited_tile_path(record)
        crop_bbox = _crop_bbox(record)
        if not before_path or not after_path or not crop_bbox:
            continue
        before = cv2.imread(str(_resolve_path(before_path, data_root)))
        after = cv2.imread(str(_resolve_path(after_path, data_root)))
        if before is None or after is None:
            continue
        x, y, w, h = [int(v) for v in crop_bbox]
        mask = _suppress_tile_border(diff_mask(before, after), record)
        if after.shape[1] != w or after.shape[0] != h:
            after = cv2.resize(after, (w, h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        end_x = min(width, x + w)
        end_y = min(height, y + h)
        eff_w = end_x - x
        eff_h = end_y - y
        if eff_w <= 0 or eff_h <= 0:
            continue
        mask_f = (mask[:eff_h, :eff_w] > 0).astype(np.float32)[..., None]
        sum_buffer[y:end_y, x:end_x] += after[:eff_h, :eff_w].astype(np.float32) * mask_f
        weight_buffer[y:end_y, x:end_x] += mask_f
        used_records.append(record)

    if not used_records:
        raise RuntimeError("No valid single-operation records were composed.")
    final = original.astype(np.float32)
    valid = weight_buffer > 0
    averaged = np.divide(sum_buffer, weight_buffer, out=np.zeros_like(sum_buffer), where=valid)
    np.copyto(final, averaged, where=np.repeat(valid, 3, axis=2))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), np.clip(final, 0, 255).astype(np.uint8))
    return {
        "source_image": str(original_image_path),
        "target_image": str(output_path),
    }


def _select_records(
    records: list[dict[str, Any]],
    geometry_table: dict[str, dict[str, Any]],
    image_id: str,
    operations: list[str],
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    if ":" not in image_id:
        raise ValueError("--image-id must be formatted as group:index_in_group")
    group, index = image_id.split(":", 1)
    by_person = [r for r in records if _person_key(r) == (group, index)]
    if not by_person:
        raise KeyError(f"No records found for image id {image_id}")
    desired = [parse_operation_spec(op) for op in operations]
    selected = []
    for params, level in desired:
        desired_id = f"{'+'.join(params)}:{level}"
        match = next((r for r in by_person if _operation_id(r) == desired_id), None)
        if match is None:
            available = sorted({_operation_id(r) for r in by_person})
            raise KeyError(f"Missing operation {params}:{level} for {image_id}. Available examples: {available[:20]}")
        selected.append(match)
    geometry = geometry_table.get(image_id)
    if geometry is None:
        geometry = next((record for record in by_person if _source_image_path(record)), {})
    source_path = _geometry_source_image_path(geometry) or _source_image_path(by_person[0])
    if source_path is None:
        raise KeyError(f"Record for {image_id} does not contain a source image path.")
    return source_path, geometry, selected


def _is_body_record(record: dict[str, Any]) -> bool:
    if str(record.get("subset", "")).lower() == "body":
        return True
    params, _ = _operation_params_and_level(record)
    return bool(params & BODY_EDIT_PARAMS)


def _geometry_record(geometry: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    if "encoder_image_process_info" in geometry:
        return geometry
    for record in records:
        if "encoder_image_process_info" in record:
            return record
    raise KeyError("Selected professional sample does not contain encoder_image_process_info.")


def parse_operation_specs(args: argparse.Namespace) -> list[str]:
    specs = []
    if args.operations:
        specs.extend(item.strip() for item in args.operations.split(",") if item.strip())
    if not specs:
        raise ValueError("Provide --operations op1:level[,op2:level...].")
    return specs


def save_professional_augmentations(
    source: np.ndarray,
    target: np.ndarray,
    geometry_record: dict[str, Any],
    is_body_edit: bool,
    output_dir: str | Path,
    count: int | None,
    seed: int,
) -> list[dict[str, str]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if source.shape[:2] != target.shape[:2]:
        target = cv2.resize(target, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_LINEAR)

    specs = generate_professional_augmentation_specs(geometry_record, (source.shape[1], source.shape[0]), is_body_edit, seed)
    if count is not None:
        specs = specs[:count]
    pairs = []
    for idx, spec in enumerate(specs):
        width, height = spec["target_resolution"]
        pair = save_warped_pair(
            source,
            target,
            spec,
            output_dir,
            prefix=f"professional_{idx:03d}_{width}x{height}",
            rotate=True,
            do_flip=False,
        )
        pair["target_resolution"] = [int(width), int(height)]
        pair["rotation"] = int(spec.get("rotation", 0))
        pairs.append(pair)
    return pairs


def run_compose(args: argparse.Namespace) -> None:
    records = _records_from_json(args.single_op_json)
    geometry_table = _load_person_geometry(args.single_op_json, args.data_root, args.person_geometry_json)
    original_path, geometry, selected = _select_records(records, geometry_table, args.image_id, parse_operation_specs(args))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stale_manifest = output_dir / "manifest.json"
    if stale_manifest.exists() or stale_manifest.is_symlink():
        stale_manifest.unlink()
    full_dir = output_dir / "full_resolution"
    full_dir.mkdir(parents=True, exist_ok=True)
    source_path = _resolve_path(original_path, args.data_root)
    full_target_path = full_dir / "target.png"
    compose_records(source_path, selected, full_target_path, args.data_root)
    source = cv2.imread(str(source_path))
    target = cv2.imread(str(full_target_path))
    if source is None:
        raise FileNotFoundError(f"Could not read source image: {source_path}")
    if target is None:
        raise FileNotFoundError(f"Could not read target image: {full_target_path}")
    full_source_path = full_dir / "source.png"
    cv2.imwrite(str(full_source_path), source)

    geom = _geometry_record(geometry, selected)
    save_warped_pair(
        source,
        target,
        geom["encoder_image_process_info"],
        output_dir / "extractor_2k",
        prefix="extractor_2k",
        rotate=False,
        do_flip=False,
    )
    qwen_pairs = save_professional_augmentations(
        source,
        target,
        geom,
        any(_is_body_record(record) for record in selected),
        output_dir / "qwen_augmentations",
        count=args.max_augmentations,
        seed=args.seed,
    )
    if not qwen_pairs:
        raise RuntimeError("No valid Qwen-image augmentation pairs were generated for this sample.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Construct MirrorPPR professional subset samples from released operation tiles.")
    parser.add_argument("--single-op-json", required=True)
    parser.add_argument("--data-root", default=None, help="Root of the downloaded MirrorPPR47M dataset release.")
    parser.add_argument("--person-geometry-json", default=None, help="Optional person_geometry.json. Defaults to the sibling of --single-op-json.")
    parser.add_argument("--image-id", required=True, help="group:index_in_group")
    parser.add_argument("--operations", required=True, help="Comma-separated operations, e.g. high_mouth:100,body_shape_thin_shoulders:-100.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-augmentations", type=int, default=None, help="Optional cap. By default all augmentations generated by the original resolution policy are saved.")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_compose(args)


if __name__ == "__main__":
    main()
