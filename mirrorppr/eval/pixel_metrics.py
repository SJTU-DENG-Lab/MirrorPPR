from __future__ import annotations

import argparse
import os
from pathlib import Path

from mirrorppr.eval.evaluate import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PSNR, SSIM, and LPIPS from MirrorPPR inference_log.json.")
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-home", default=None, help="Optional TORCH_HOME for offline LPIPS/VGG weights.")
    args = parser.parse_args()
    if args.torch_home:
        os.environ["TORCH_HOME"] = args.torch_home
    output = args.output_file or str(Path(args.log_file).with_name("pixel_metrics.json"))
    evaluate(args.log_file, output, metrics={"pixel"}, device=args.device)


if __name__ == "__main__":
    main()
