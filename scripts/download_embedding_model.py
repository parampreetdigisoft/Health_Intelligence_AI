#!/usr/bin/env python3
"""
Download all-MiniLM-L6-v2 for offline embedding use.

Run once on a machine that can reach Hugging Face, then copy the output
folder to the server (or run this script on the server if network allows).

Usage (from Health_Intelligence_AI folder):
    python scripts/download_embedding_model.py
"""
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR = ROOT / "models" / "all-MiniLM-L6-v2"
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


def main() -> None:
    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODEL_ID} ...")
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(OUTPUT_DIR),
    )

    if not any((OUTPUT_DIR / name).is_file() for name in WEIGHT_FILES):
        raise RuntimeError(
            f"Download finished but no model weights found in {OUTPUT_DIR}. "
            "Check network/proxy access to huggingface.co."
        )

    print(f"Saved to: {OUTPUT_DIR}")
    print()
    print("On the server, either:")
    print("  1. Copy this folder to the same path under Health_Intelligence_AI, or")
    print("  2. Set EMBEDDING_MODEL_PATH in .env to the folder's absolute path")


if __name__ == "__main__":
    main()
