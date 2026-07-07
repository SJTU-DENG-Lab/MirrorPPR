from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def load_rows(log_file: str | Path) -> list[dict[str, Any]]:
    with Path(log_file).open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"{log_file} must contain a JSON list.")
    return rows


def base_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "test_index": row.get("test_index"),
        "edit_instruction": row.get("edit_instruction", ""),
        "pair2_original_path": row.get("pair2_original_path"),
        "pair2_edited_path": row.get("pair2_edited_path"),
        "output_image_path": row.get("output_image_path"),
    }


def evaluate_pixel_metrics(rows: list[dict[str, Any]], device: str) -> tuple[dict[str, Any], dict[Any, dict[str, Any]]]:
    import torchvision.transforms as transforms
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(device)
    to_tensor = transforms.ToTensor()
    results: dict[Any, dict[str, Any]] = {}

    for row in tqdm(rows, desc="Pixel metrics"):
        pred_path = row.get("output_image_path")
        gt_path = row.get("pair2_edited_path")
        if not pred_path or not gt_path or not Path(pred_path).exists() or not Path(gt_path).exists():
            continue

        pred = Image.open(pred_path).convert("RGB")
        gt = Image.open(gt_path).convert("RGB")
        if pred.size != gt.size:
            pred = pred.resize(gt.size, Image.Resampling.BILINEAR)

        pred_t = to_tensor(pred).unsqueeze(0).to(device)
        gt_t = to_tensor(gt).unsqueeze(0).to(device)
        pred_l = pred_t * 2.0 - 1.0
        gt_l = gt_t * 2.0 - 1.0

        psnr_val = psnr(pred_t, gt_t)
        ssim_val = ssim(pred_t, gt_t)
        lpips_val = lpips(pred_l, gt_l)
        results[row.get("test_index")] = {
            "psnr": float(psnr_val.item()),
            "ssim": float(ssim_val.item()),
            "lpips": float(lpips_val.item()),
        }

    count = len(results)
    summary = {
        "valid_pixel_samples": count,
        "average_psnr": float(psnr.compute().item()) if count else None,
        "average_ssim": float(ssim.compute().item()) if count else None,
        "average_lpips": float(lpips.compute().item()) if count else None,
    }
    return summary, results


def detect_faces_with_padding(app, image: np.ndarray):
    faces = app.get(image)
    if faces:
        return faces
    h, w = image.shape[:2]
    pad_h, pad_w = int(h * 0.25), int(w * 0.25)
    padded = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return app.get(padded)


def validate_insightface_root(model_root: str | Path) -> None:
    model_dir = Path(model_root) / "models" / "buffalo_l"
    required = ["1k3d68.onnx", "2d106det.onnx", "det_10g.onnx", "genderage.onnx", "w600k_r50.onnx"]
    missing = [name for name in required if not (model_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing InsightFace buffalo_l files under {model_dir}: {missing}. "
            "Download buffalo_l.zip and unpack it so that models/buffalo_l contains the ONNX files."
        )


def validate_onnxruntime_provider(ctx_id: int) -> None:
    import onnxruntime as ort

    providers = ort.get_available_providers()
    if ctx_id >= 0 and "CUDAExecutionProvider" not in providers:
        raise RuntimeError(
            "InsightFace identity metrics were requested on GPU, but ONNX Runtime cannot see CUDAExecutionProvider. "
            f"Available providers: {providers}. Install onnxruntime-gpu in this environment, or pass "
            "--identity-ctx-id -1 to run identity metrics on CPU intentionally."
        )


def evaluate_identity_metrics(
    rows: list[dict[str, Any]],
    model_root: str | Path,
    det_size: int,
    ctx_id: int,
) -> tuple[dict[str, Any], dict[Any, dict[str, Any]]]:
    from insightface.app import FaceAnalysis

    validate_insightface_root(model_root)
    validate_onnxruntime_provider(ctx_id)
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"insightface\.utils\.transform")
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"insightface\.utils\.face_align")
    app = FaceAnalysis(name="buffalo_l", root=str(model_root))
    app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
    results: dict[Any, dict[str, Any]] = {}

    for row in tqdm(rows, desc="Identity similarity"):
        edit_path = row.get("pair2_edited_path")
        pred_path = row.get("output_image_path")
        if not edit_path or not pred_path:
            continue

        images = [cv2.imread(str(path)) for path in [edit_path, pred_path]]
        if any(image is None for image in images):
            continue

        faces = [detect_faces_with_padding(app, image) for image in images]
        if any(len(face) == 0 for face in faces):
            continue

        emb_edit, emb_pred = [face[0].embedding for face in faces]
        results[row.get("test_index")] = {
            "identity_similarity_edited": cosine(emb_edit, emb_pred),
        }

    count = len(results)
    summary = {
        "valid_identity_samples": count,
        "average_identity_similarity_edited": (
            float(np.mean([item["identity_similarity_edited"] for item in results.values()])) if count else None
        ),
    }
    return summary, results


def merge_results(
    rows: list[dict[str, Any]],
    pixel_results: dict[Any, dict[str, Any]] | None = None,
    identity_results: dict[Any, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    pixel_results = pixel_results or {}
    identity_results = identity_results or {}
    merged = []
    for row in rows:
        test_index = row.get("test_index")
        item = base_record(row)
        item.update(pixel_results.get(test_index, {}))
        item.update(identity_results.get(test_index, {}))
        merged.append(item)
    return merged


def normalize_metrics(metrics: list[str]) -> set[str]:
    selected = set(metrics)
    if "all" in selected:
        return {"pixel", "identity"}
    return selected


def evaluate(
    log_file: str | Path,
    output_file: str | Path,
    metrics: set[str],
    insightface_root: str | Path | None = None,
    det_size: int = 640,
    identity_ctx_id: int | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    import torch

    rows = load_rows(log_file)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    identity_ctx_id = identity_ctx_id if identity_ctx_id is not None else (0 if torch.cuda.is_available() else -1)

    summary: dict[str, Any] = {"total_samples": len(rows)}
    pixel_results: dict[Any, dict[str, Any]] = {}
    identity_results: dict[Any, dict[str, Any]] = {}

    if "pixel" in metrics:
        pixel_summary, pixel_results = evaluate_pixel_metrics(rows, device=device)
        summary.update(pixel_summary)

    if "identity" in metrics:
        if insightface_root is None:
            raise ValueError("--insightface-root is required when identity metrics are enabled.")
        identity_summary, identity_results = evaluate_identity_metrics(
            rows,
            model_root=insightface_root,
            det_size=det_size,
            ctx_id=identity_ctx_id,
        )
        summary.update(identity_summary)

    report = {
        "summary": summary,
        "individual_results": merge_results(rows, pixel_results, identity_results),
    }
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate MirrorPPR inference outputs.")
    parser.add_argument("--log-file", required=True, help="Path to inference_log.json.")
    parser.add_argument("--output-file", default=None, help="Path to save the combined metrics JSON.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["all"],
        choices=["all", "pixel", "identity"],
        help="Metrics to compute. Use 'all' for pixel and identity metrics.",
    )
    parser.add_argument(
        "--insightface-root",
        "--model-root",
        dest="insightface_root",
        default=None,
        help="Directory containing models/buffalo_l for InsightFace identity metrics.",
    )
    parser.add_argument("--det-size", type=int, default=640, help="InsightFace detector input size.")
    parser.add_argument("--identity-ctx-id", type=int, default=None, help="InsightFace ctx_id. Use -1 for CPU.")
    parser.add_argument("--device", default=None, help="Torch device for pixel metrics, for example cuda or cpu.")
    parser.add_argument("--torch-home", default=None, help="Optional TORCH_HOME for offline LPIPS/VGG weights.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.torch_home:
        os.environ["TORCH_HOME"] = args.torch_home
    output_file = args.output_file or str(Path(args.log_file).with_name("evaluation_metrics.json"))
    evaluate(
        log_file=args.log_file,
        output_file=output_file,
        metrics=normalize_metrics(args.metrics),
        insightface_root=args.insightface_root,
        det_size=args.det_size,
        identity_ctx_id=args.identity_ctx_id,
        device=args.device,
    )


if __name__ == "__main__":
    main()
