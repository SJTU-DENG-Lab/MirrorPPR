from __future__ import annotations

import argparse
from pathlib import Path

from mirrorppr.eval.evaluate import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate identity preservation with InsightFace ArcFace.")
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--model-root", "--insightface-root", dest="insightface_root", required=True)
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument("--identity-ctx-id", type=int, default=None)
    args = parser.parse_args()
    output = args.output_file or str(Path(args.log_file).with_name("identity_similarity.json"))
    evaluate(
        args.log_file,
        output,
        metrics={"identity"},
        insightface_root=args.insightface_root,
        det_size=args.det_size,
        identity_ctx_id=args.identity_ctx_id,
    )


if __name__ == "__main__":
    main()
