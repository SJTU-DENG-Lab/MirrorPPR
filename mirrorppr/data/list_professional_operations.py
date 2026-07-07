from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mirrorppr.data.io import read_json, write_json


def resolve_path(path: str | Path, data_root: str | Path | None) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return Path(data_root) / path if data_root else path


def executable(record: dict[str, Any], data_root: str | Path | None, validate_files: bool) -> bool:
    if not record.get("operation_id") or not record.get("crop_bbox_in_original"):
        return False
    if not validate_files:
        return True
    for key in ("source_tile_path", "edited_tile_path"):
        value = record.get(key)
        if not value or not resolve_path(value, data_root).exists():
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="List executable MirrorPPR47M professional operations for one image.")
    parser.add_argument("--single-op-json", required=True)
    parser.add_argument("--data-root", default=None, help="Root of the downloaded MirrorPPR47M dataset release.")
    parser.add_argument("--image-id", required=True, help="group:index_in_group")
    parser.add_argument("--subset", choices=["all", "face", "body"], default="all")
    parser.add_argument("--no-validate-files", action="store_true", help="Do not check whether referenced tile files exist.")
    parser.add_argument("--output-json", default=None, help="Defaults to <image_id>_professional_operations.json.")
    args = parser.parse_args()

    rows = read_json(args.single_op_json)
    if not isinstance(rows, list):
        raise TypeError(f"{args.single_op_json} must contain a list.")

    selected = []
    for record in rows:
        if str(record.get("image_id")) != args.image_id:
            continue
        subset = str(record.get("subset", ""))
        if args.subset != "all" and subset != args.subset:
            continue
        if executable(record, args.data_root, not args.no_validate_files):
            selected.append(
                {
                    "image_id": args.image_id,
                    "subset": subset,
                    "operation_id": str(record["operation_id"]),
                }
            )

    selected.sort(key=lambda item: (item["subset"], item["operation_id"]))
    grouped: dict[str, list[str]] = {"face": [], "body": []}
    for item in selected:
        grouped.setdefault(item["subset"], []).append(item["operation_id"])

    print(f"image_id: {args.image_id}")
    print(f"total executable operations: {len(selected)}")
    for subset in ["face", "body"]:
        if args.subset != "all" and args.subset != subset:
            continue
        operations = grouped.get(subset, [])
        print(f"{subset}: {len(operations)}")
        for op in operations:
            print(f"  {op}")

    output_json = args.output_json or f"{args.image_id.replace(':', '_')}_professional_operations.json"
    write_json(
        output_json,
        {
            "image_id": args.image_id,
            "description": "These are executable atomic operations for this image. Any compatible subset can be composed by passing comma-separated operation_id values to professional_subset.py --operations.",
            "atomic_operations": selected,
            "by_subset": grouped,
            "usage": {
                "compose_with": "python mirrorppr/data/professional_subset.py --operations op1:level,op2:level",
                "example_operations_arg": ",".join(item["operation_id"] for item in selected[:2]),
            },
        },
    )
    print(f"Saved operation list to: {output_json}")


if __name__ == "__main__":
    main()
