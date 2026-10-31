"""Run scan.py on Modal GPUs, splitting a layer's neurons across containers.

Results land in the same results/scans/L{layer}_dose{dose}.jsonl as a local scan, and
neurons already in that file are skipped.

    uv run --extra modal modal run modal_scan.py --layer 20 --dose -1
    uv run --extra modal modal run modal_scan.py --layer 20 --gpus 16
"""

import json

import modal

app = modal.App("qwen-steering")

hf_cache = modal.Volume.from_name("qwen-steering-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.15.0", "transformers==5.21.1", "accelerate", "safetensors", "flash-linear-attention")
    .env({"HF_HOME": "/hf"})
    .add_local_python_source("qwen", "neuron", "scan")
)


@app.function(image=image, gpu=["L4", "A10", "L40S"], volumes={"/hf": hf_cache}, timeout=3600)
def scan_slice(neurons: list[int], layer: int, dose: float, batch: int, tokens: int) -> list[dict]:
    import time

    import torch

    from qwen import load
    from scan import scan_neurons

    tok, model, layers = load()
    hf_cache.commit()
    start = time.time()
    rows = list(scan_neurons(tok, model, layers, layer, dose, neurons, batch, tokens))
    rate = len(rows) / (time.time() - start)
    print(f"{torch.cuda.get_device_name()}: {len(rows)} neurons, {rate:.1f}/s")
    return rows


@app.local_entrypoint()
def main(layer: int = 5, dose: float = -1.0, gpus: int = 8, batch: int = 128, tokens: int = 48, limit: int = 0):
    from scan import scan_path

    out_path = scan_path(layer, dose)
    done = set()
    if out_path.exists():
        done = {json.loads(l)["neuron"] for l in out_path.open()}
    n_neurons = limit or 9216
    todo = [n for n in range(n_neurons) if n not in done]
    if not todo:
        print(f"{out_path.name} is complete")
        return

    size = -(-len(todo) // gpus)
    slices = [todo[i : i + size] for i in range(0, len(todo), size)]
    print(f"scanning {len(todo)} neurons of layer {layer} on {len(slices)} GPUs")
    with out_path.open("a") as f:
        for rows in scan_slice.map(slices, kwargs={"layer": layer, "dose": dose, "batch": batch, "tokens": tokens}):
            for row in rows:
                f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"slice done: {len(rows)} neurons")
