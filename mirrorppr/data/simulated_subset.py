from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mirrorppr.data.ffhq_wild_blend import (
    FFHQ_SHIFT_DIRECTIONS,
    FFHQWildFrameAugmentor,
    blend_aligned_face_into_wild,
    crop_aligned_from_quad,
    load_wild_info,
)
from mirrorppr.data.image_ops import bgr_to_pil, load_rgb
from mirrorppr.data.io import read_json
from mirrorppr.data.llw_face_editor import LLWFaceRetoucher, normalize_operations


def parse_operation_specs(args: argparse.Namespace) -> list[dict[str, object]]:
    specs = []
    if args.operations:
        specs.extend(item.strip() for item in args.operations.split(",") if item.strip())
    if not specs:
        raise ValueError("Provide --operations name:strength[,name:strength...].")
    return normalize_operations(specs)


def resolve_wild_info(args: argparse.Namespace) -> dict[str, object]:
    if args.wild_info_json:
        data = read_json(args.wild_info_json)
        if not isinstance(data, dict) or "face_quad" not in data:
            raise ValueError("--wild-info-json must contain a JSON object with face_quad.")
        return data
    if args.ffhq_metadata and args.image_id:
        return load_wild_info(args.ffhq_metadata, args.image_id)
    raise ValueError("Provide crop information with --ffhq-metadata plus --image-id, or --wild-info-json.")


def save_wild_frame_augmentations(
    source_wild,
    target_wild,
    wild_info: dict[str, object],
    output_dir: Path,
    count: int,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    augmentor = FFHQWildFrameAugmentor(wild_info, wild_size=source_wild.size)
    directions = list(FFHQ_SHIFT_DIRECTIONS.items())[:count]
    for idx, (direction_name, direction_vec) in enumerate(directions):
        shifted_quad = augmentor.get_shifted_quad(direction_vec, magnitude_ratio=1.0)
        source_out = output_dir / f"simulated_{idx:03d}_{direction_name}_source.png"
        target_out = output_dir / f"simulated_{idx:03d}_{direction_name}_target.png"
        crop_aligned_from_quad(source_wild, shifted_quad).save(source_out)
        crop_aligned_from_quad(target_wild, shifted_quad).save(target_out)
    return len(directions)


def run_single(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stale_manifest = output_dir / "manifest.json"
    if stale_manifest.exists() or stale_manifest.is_symlink():
        stale_manifest.unlink()
    if args.num_augmentations > 0 and not args.wild_image:
        raise ValueError("Faithful simulated self-augmentation requires --wild-image and crop information. Use --num-augmentations 0 to only save the aligned pair.")

    operations = parse_operation_specs(args)
    retoucher = LLWFaceRetoucher()
    edited_bgr = retoucher.apply_operations(args.aligned_image, operations)
    edited_aligned = bgr_to_pil(edited_bgr)
    aligned = load_rgb(args.aligned_image)
    aligned_source_out = output_dir / "aligned_source.png"
    aligned_target_out = output_dir / "aligned_target.png"
    aligned.save(aligned_source_out)
    edited_aligned.save(aligned_target_out)

    if args.wild_image:
        wild = load_rgb(args.wild_image)
        wild_info = resolve_wild_info(args)
        face_quad = wild_info["face_quad"]
        wild_source = blend_aligned_face_into_wild(aligned, wild, face_quad)
        wild_source_out = output_dir / "wild_source.png"
        wild_source.save(wild_source_out)
        wild_retouched = blend_aligned_face_into_wild(edited_aligned, wild, face_quad)
        wild_target_out = output_dir / "wild_target.png"
        wild_retouched.save(wild_target_out)

        if args.num_augmentations > 0:
            save_wild_frame_augmentations(
                wild_source,
                wild_retouched,
                wild_info,
                output_dir / "augmentations",
                min(args.num_augmentations, len(FFHQ_SHIFT_DIRECTIONS)),
            )
    elif args.num_augmentations > 0:
        raise ValueError("Internal error: wild inputs are required before saving simulated augmentations.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Construct MirrorPPR simulated subset samples.")
    parser.add_argument("--aligned-image", required=True, help="1024x1024 aligned FFHQ image.")
    parser.add_argument(
        "--operations",
        default=None,
        help="Comma-separated operations as name:strength, e.g. eye_resize:100,nose_alar:-100.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wild-image", default=None)
    parser.add_argument("--ffhq-metadata", default=None)
    parser.add_argument("--image-id", default=None)
    parser.add_argument("--wild-info-json", default=None, help="JSON file with face_quad and optional face_rect/pixel_size.")
    parser.add_argument("--num-augmentations", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_single(args)


if __name__ == "__main__":
    main()
