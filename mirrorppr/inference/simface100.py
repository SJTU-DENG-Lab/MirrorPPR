from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mirrorppr.data.image_ops import round_to_multiple


def _glob_required(pattern: str) -> list[str]:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {pattern}")
    return files


def _default_paths(weights_root: Path, qwen_root: Path | None) -> dict[str, Any]:
    qwen = qwen_root or weights_root / "qwen_image_edit"
    face = weights_root / "mirrorppr_face"
    return {
        "dit": _glob_required(str(qwen / "transformer" / "diffusion_pytorch_model*.safetensors")),
        "text_encoder": _glob_required(str(qwen / "text_encoder" / "model*.safetensors")),
        "vae": str(qwen / "vae" / "diffusion_pytorch_model.safetensors"),
        "processor": str(qwen / "processor"),
        "mae": str(face / "mae" / "mae_pretrained.safetensors"),
        "rformer": str(face / "rformer" / "rformer.safetensors"),
        "connector": str(face / "connector" / "connector.safetensors"),
        "lora": str(face / "lora" / "lora.safetensors"),
    }


def load_mirrorppr_face(paths: dict[str, Any], device: str = "cuda"):
    import torch
    from diffsynth import load_state_dict
    from diffsynth.pipelines.qwen_image import ModelConfig, QwenImagePipeline

    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(path=paths["dit"]),
            ModelConfig(path=paths["text_encoder"]),
            ModelConfig(path=paths["vae"]),
            ModelConfig(path=paths["mae"]),
            ModelConfig(path=paths["rformer"]),
            ModelConfig(path=paths["connector"]),
        ],
        tokenizer_config=None,
        processor_config=ModelConfig(path=paths["processor"]),
    )
    if pipe.rformer is None:
        raise RuntimeError("The MirrorPPR runtime did not load the R-Former module.")
    if not hasattr(pipe, "connector") or pipe.connector is None:
        raise RuntimeError("The MirrorPPR runtime did not load the connector module.")
    pipe.rformer.load_state_dict(load_state_dict(paths["rformer"]))
    pipe.connector.load_state_dict(load_state_dict(paths["connector"]))
    pipe.load_lora(pipe.dit, paths["lora"])
    return pipe


def _dataset_image(root: Path, item: dict[str, Any], key: str, resolution: str) -> Image.Image:
    suffix = "" if resolution == "base" else f"_{resolution}"
    resolved_key = key + suffix if key + suffix in item else key
    return Image.open(root / item[resolved_key]).convert("RGB")


def _output_name(index: int, item: dict[str, Any]) -> str:
    p1 = Path(item["pair1_origin"]).stem
    p2 = Path(item["pair2_origin"]).stem
    return f"mirrorppr_face_idx{index:04d}_p1_{p1}_p2_{p2}.jpg"


def run_worker(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with Path(args.dataset_json).open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)
    chunk = dataset[args.start : min(args.end, len(dataset))]
    paths = _default_paths(Path(args.weights_root), Path(args.qwen_root) if args.qwen_root else None)
    pipe = load_mirrorppr_face(paths)
    log = []
    for offset, item in enumerate(tqdm(chunk, desc=f"GPU {args.gpu_id}")):
        index = args.start + offset
        example_origin = _dataset_image(dataset_root, item, "pair1_origin", args.resolution)
        example_target = _dataset_image(dataset_root, item, "pair1_edited", args.resolution)
        query = _dataset_image(dataset_root, item, "pair2_origin", args.resolution)
        target = dataset_root / item["pair2_edited"]
        width, height = query.size
        width = round_to_multiple(width, 16)
        height = round_to_multiple(height, 16)
        output_path = output_dir / _output_name(index, item)
        result = pipe(
            "",
            example_origin=example_origin,
            example_target=example_target,
            edit_image=query,
            seed=args.seed,
            num_inference_steps=args.steps,
            height=height,
            width=width,
            edit_image_auto_resize=False,
        )
        result.save(output_path)
        log.append(
            {
                "test_index": index,
                "pair1_original_path": str(dataset_root / item["pair1_origin"]),
                "pair1_edited_path": str(dataset_root / item["pair1_edited"]),
                "pair2_original_path": str(dataset_root / item["pair2_origin"]),
                "pair2_edited_path": str(target),
                "output_image_path": str(output_path),
                "edit_instruction": item.get("edit_instruction", ""),
            }
        )
    shard_log = output_dir / f"inference_log_{args.start}_{args.end}.json"
    with shard_log.open("w", encoding="utf-8") as handle:
        json.dump(log, handle, ensure_ascii=False, indent=2)


def merge_logs(output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(output_dir.glob("inference_log_*_*.json")):
        with path.open("r", encoding="utf-8") as handle:
            rows.extend(json.load(handle))
    if not rows:
        raise RuntimeError(f"No worker logs were found in {output_dir}. Check worker stderr above.")
    rows.sort(key=lambda row: row.get("test_index", 0))
    with (output_dir / "inference_log.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
    print(f"Merged {len(rows)} records into {output_dir / 'inference_log.json'}")


def run_master(args: argparse.Namespace) -> None:
    dataset_json = Path(args.dataset_json or Path(args.dataset_root) / "dataset.json")
    with dataset_json.open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)
    if args.limit is not None:
        total = min(args.limit, len(dataset))
    else:
        total = len(dataset)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()] if args.gpus else ["0"]
    chunk_size = math.ceil(total / len(gpus))
    procs = []
    for local_idx, gpu in enumerate(gpus):
        start = local_idx * chunk_size
        end = min(total, (local_idx + 1) * chunk_size)
        if start >= end:
            continue
        cmd = [
            sys.executable,
            "-m",
            "mirrorppr.inference.simface100",
            "--worker",
            "--gpu-id",
            gpu,
            "--start",
            str(start),
            "--end",
            str(end),
            "--dataset-root",
            args.dataset_root,
            "--dataset-json",
            str(dataset_json),
            "--weights-root",
            args.weights_root,
            "--output-dir",
            args.output_dir,
            "--resolution",
            args.resolution,
            "--steps",
            str(args.steps),
            "--seed",
            str(args.seed),
        ]
        if args.qwen_root:
            cmd += ["--qwen-root", args.qwen_root]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        procs.append(subprocess.Popen(cmd, env=env))
    codes = [proc.wait() for proc in procs]
    if any(code != 0 for code in codes):
        try:
            merge_logs(args.output_dir)
        except RuntimeError:
            pass
        raise SystemExit(f"One or more workers failed: {codes}")
    merge_logs(args.output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MirrorPPR-Face on SimFace-100.")
    parser.add_argument("--dataset-root", required=True, help="Directory containing dataset.json and images/.")
    parser.add_argument("--dataset-json", default=None)
    parser.add_argument("--weights-root", required=True, help="Prepared MirrorPPR-Face weight directory.")
    parser.add_argument("--qwen-root", default=None, help="Optional external Qwen-Image-Edit directory.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resolution", choices=["base", "2k"], default="base")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--gpus", default="0", help="Comma-separated physical GPU ids for master mode.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.dataset_json is None:
        args.dataset_json = str(Path(args.dataset_root) / "dataset.json")
    if args.worker:
        run_worker(args)
    else:
        run_master(args)


if __name__ == "__main__":
    main()
