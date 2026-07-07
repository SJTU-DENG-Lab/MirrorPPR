from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random

from PIL import Image


@dataclass
class AugmentationSpec:
    width: int
    height: int
    crop_box: tuple[int, int, int, int]
    rotation: int = 0
    flip: bool = False


def _fit_crop_box(
    image_size: tuple[int, int],
    aspect_ratio: float,
    scale: float,
    required_box: tuple[int, int, int, int] | None,
    rng: random.Random,
) -> tuple[int, int, int, int]:
    width, height = image_size
    crop_area = width * height * scale
    crop_w = int((crop_area * aspect_ratio) ** 0.5)
    crop_h = int(crop_w / aspect_ratio)
    if crop_w > width:
        crop_w = width
        crop_h = int(crop_w / aspect_ratio)
    if crop_h > height:
        crop_h = height
        crop_w = int(crop_h * aspect_ratio)
    crop_w = max(16, min(width, crop_w))
    crop_h = max(16, min(height, crop_h))

    if required_box is None:
        left = rng.randint(0, max(0, width - crop_w))
        top = rng.randint(0, max(0, height - crop_h))
        return left, top, left + crop_w, top + crop_h

    rx, ry, rw, rh = required_box
    min_left = max(0, rx + rw - crop_w)
    max_left = min(rx, width - crop_w)
    min_top = max(0, ry + rh - crop_h)
    max_top = min(ry, height - crop_h)
    if min_left > max_left or min_top > max_top:
        return 0, 0, width, height
    left = rng.randint(int(min_left), int(max_left))
    top = rng.randint(int(min_top), int(max_top))
    return left, top, left + crop_w, top + crop_h


def generate_specs(
    image_size: tuple[int, int],
    count: int,
    seed: int = 0,
    output_area: int = 1024 * 1024,
    aspect_ratio_range: tuple[float, float] = (1 / 3, 3),
    required_box: tuple[int, int, int, int] | None = None,
) -> list[AugmentationSpec]:
    rng = random.Random(seed)
    specs = []
    angles = [-15, -10, -5, 0, 5, 10, 15]
    for _ in range(count):
        ar = rng.uniform(*aspect_ratio_range)
        scale = rng.uniform(0.55, 1.0)
        crop_box = _fit_crop_box(image_size, ar, scale, required_box, rng)
        target_w = int((output_area * ar) ** 0.5)
        target_h = int(target_w / ar)
        target_w = max(16, (target_w // 16) * 16)
        target_h = max(16, (target_h // 16) * 16)
        specs.append(
            AugmentationSpec(
                width=target_w,
                height=target_h,
                crop_box=crop_box,
                rotation=rng.choice(angles),
                flip=bool(rng.getrandbits(1)),
            )
        )
    return specs


def apply_spec(image: Image.Image, spec: AugmentationSpec) -> Image.Image:
    out = image.convert("RGB")
    if spec.flip:
        out = out.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    out = out.crop(spec.crop_box)
    if spec.rotation:
        out = out.rotate(spec.rotation, resample=Image.Resampling.BICUBIC, expand=False)
    return out.resize((spec.width, spec.height), Image.Resampling.LANCZOS)


def save_augmented_pair(
    source: Image.Image,
    target: Image.Image,
    output_dir: str | Path,
    count: int,
    seed: int = 0,
    prefix: str = "aug",
    required_box: tuple[int, int, int, int] | None = None,
) -> list[dict[str, object]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = generate_specs(source.size, count=count, seed=seed, required_box=required_box)
    pairs = []
    for idx, spec in enumerate(specs):
        source_out = output_dir / f"{prefix}_{idx:03d}_source.png"
        target_out = output_dir / f"{prefix}_{idx:03d}_target.png"
        apply_spec(source, spec).save(source_out)
        apply_spec(target, spec).save(target_out)
        pairs.append(
            {
                "source": str(source_out),
                "target": str(target_out),
                "augmentation": asdict(spec),
            }
        )
    return pairs
