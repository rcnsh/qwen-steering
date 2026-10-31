"""Download the public Qwen3.5-4B neuron atlas into data/atlas/.

From the Hugging Face dataset nikhilxvytla/neuron-atlas (ODC-BY):
  sweep.parquet   every neuron pushed at -0.8 and +0.8 on two prompts, 32 tokens
  labels.parquet  LLM labels for about 9,900 neurons
  meta.json       per-layer median residual norms used for doses

    uv run fetch_atlas.py
"""

import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "nikhilxvytla/neuron-atlas"
FILES = {
    "Qwen__Qwen3.5-4B/sweep/steer.parquet": "sweep.parquet",
    "Qwen__Qwen3.5-4B/sweep/meta.json": "meta.json",
    "Qwen__Qwen3.5-4B/labels.parquet": "labels.parquet",
}
ATLAS_DIR = Path(__file__).parent / "data" / "atlas"


def main():
    ATLAS_DIR.mkdir(parents=True, exist_ok=True)
    for remote, local in FILES.items():
        src = hf_hub_download(REPO, remote, repo_type="dataset")
        shutil.copyfile(src, ATLAS_DIR / local)
        print(f"{local}: {(ATLAS_DIR / local).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
