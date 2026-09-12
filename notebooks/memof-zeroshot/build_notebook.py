"""Build the Kaggle MEMFOF zero-shot inference notebook."""

import json
from pathlib import Path


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip() + "\n"}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip() + "\n",
    }


cells = [
    markdown(r"""
# Experiment 03 — official MEMFOF zero-shot master baseline

**Flow State · RoCo-45 · Optical Flow**

This Kaggle notebook mirrors the safeguards and artifact layout of the RAFT-Sintel master
baseline, but performs inference with the official **MEMFOF** implementation and the authors'
public **MEMFOF-Tartan-T-TSKH** checkpoint. It runs at native 1920×1080, uses three-frame
windows, writes forward and backward flow for both cameras, validates every output, and runs
the official Spring `flow_subsampling` executable.

The attached `raft-sintel-fb44381e.ckpt` is deliberately **not used**. It contains RAFT
parameters and is not architecture-compatible with MEMFOF.
"""),
    markdown(r"""
## Model and checkpoint identity

MEMFOF is the ICCV 2025 *Memory-Efficient Multi-Frame Optical Flow* model. The official
package consumes `[B, 3, 3, H, W]` images and returns both directions around the middle frame:
`flow[:, 0]` is middle→previous (`BW`) and `flow[:, 1]` is middle→next (`FW`).

The checkpoint used here is the authors' Hugging Face model
`egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH`, pinned to revision
`87b740bed2c7ea4890d98fd0f9d6ef5738254eb9`. It contains `config.json` and a 303 MB
`model.safetensors` file. TSKH means the mixed Sintel/KITTI/HD1K stage; this base checkpoint
has not been Spring-fine-tuned, so the experiment remains zero-shot on Spring.
"""),
    markdown(r"""
## 1. Freeze the experiment configuration

The paths below match the attached Kaggle datasets exactly. Internet must be enabled once to
clone the official MEMFOF source and download its official checkpoint.
"""),
    code(r"""
from pathlib import Path
import hashlib, json, os, platform, re, shutil, subprocess, sys, time

TEAM_ID, TEAM_NAME = "RoCo-45", "Flow State"
MODEL_NAME = "MEMFOF"
ITERATIONS = 8
SEQUENCE_LENGTH = 3
NUM_GPUS = 2
MAX_FORWARD_SIDE = None

MEMFOF_REPO = "https://github.com/msu-video-group/memfof.git"
MEMFOF_SOURCE_BRANCH = "dev"
MEMFOF_SOURCE_REF = "a51de9fc59c6fe20ba08e079372c7b583d58a712"
CHECKPOINT_REPO_ID = "egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH"
CHECKPOINT_REVISION = "87b740bed2c7ea4890d98fd0f9d6ef5738254eb9"

KAGGLE_INPUT = Path("/kaggle/input")
FLOW_STATE_INPUT_ROOT = KAGGLE_INPUT / "datasets/syedmohaiminulhoque"
LEFT_ROOT = FLOW_STATE_INPUT_ROOT / "test-frame-left-right/test_frame_left/spring"
RIGHT_ROOT = FLOW_STATE_INPUT_ROOT / "test-frame-left-right/test_frame_right/spring"
CAM_ROOT = FLOW_STATE_INPUT_ROOT / "test-frame-left-right/test_cam_data/spring"
RAFT_CHECKPOINT = FLOW_STATE_INPUT_ROOT / "ckpoint/raft-sintel-fb44381e.ckpt"
SUBSAMPLING_SOURCE = FLOW_STATE_INPUT_ROOT / "flow-subsampling/flow_subsampling"

WORK_ROOT = Path("/kaggle/working")
SCRATCH_ROOT = Path("/kaggle/temp")
MEMFOF_DIR = WORK_ROOT / "memfof"
SPRING_ROOT = SCRATCH_ROOT / "exp03_spring_merged"
OUTPUT_ROOT = SCRATCH_ROOT / "exp03_memfof_predictions" / "spring"
ARTIFACT_DIR = WORK_ROOT / "exp03_artifacts"
INFERENCE_SCRIPT = MEMFOF_DIR / "run_memfof_spring.py"

SESSION_START = time.monotonic()
MAX_SESSION_HOURS = 12.0
PACKAGING_RESERVE_MINUTES = 45

assert KAGGLE_INPUT.is_dir(), "Run this notebook in Kaggle."
experiment_config = {
    "team": f"{TEAM_ID} — {TEAM_NAME}",
    "model": MODEL_NAME,
    "checkpoint": CHECKPOINT_REPO_ID,
    "checkpoint_revision": CHECKPOINT_REVISION,
    "iterations": ITERATIONS,
    "sequence_length": SEQUENCE_LENGTH,
    "input_resolution": "native 1920x1080 (no rescaling)",
    "gpus": NUM_GPUS,
    "fine_tuning": False,
    "test_time_augmentation": False,
}
print(json.dumps(experiment_config, indent=2))
"""),
    markdown(r"""
## 2. Hardware and storage preflight

This follows the RAFT notebook's T4×2 and storage checks. Predictions are kept in
`/kaggle/temp`; `/kaggle/working` is reserved for the final artifact.
"""),
    code(r"""
gpu = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
    text=True, capture_output=True, check=True,
)
gpu_lines = [line.strip() for line in gpu.stdout.splitlines() if line.strip()]
print("\n".join(gpu_lines))
if len(gpu_lines) != NUM_GPUS or any("T4" not in line for line in gpu_lines):
    raise RuntimeError("Select Kaggle GPU T4 x2 and restart the session.")

for path in (WORK_ROOT, SCRATCH_ROOT):
    path.mkdir(parents=True, exist_ok=True)
    print(f"{path}: {shutil.disk_usage(path).free / 1024**3:.1f} GiB free")
if shutil.disk_usage(SCRATCH_ROOT).free / 1024**3 < 60:
    raise RuntimeError("Less than 60 GiB free in /kaggle/temp.")
print("Python:", sys.version.split()[0], "OS:", platform.platform())
"""),
    markdown(r"""
## 3. Verify and assemble the exact Spring inputs

The left frames, right frames, and camera metadata are attached as separate read-only Kaggle
trees. Symlinks assemble the canonical `spring/test/<sequence>/...` layout without copying
images. Intrinsics are retained in that layout, although optical-flow inference itself does
not use camera calibration.
"""),
    code(r"""
EXPECTED_SEQUENCES = {"0003", "0019", "0028", "0029", "0031", "0034", "0035", "0040", "0042", "0046"}

required_roots = {"left": LEFT_ROOT, "right": RIGHT_ROOT, "camera": CAM_ROOT}
missing_roots = {name: str(path) for name, path in required_roots.items() if not path.is_dir()}
if missing_roots:
    raise FileNotFoundError("Missing configured Kaggle inputs:\n" + json.dumps(missing_roots, indent=2))

def ensure_directory_link(source: Path, destination: Path):
    if not source.is_dir():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise RuntimeError(f"Conflicting symlink: {destination}")
    elif destination.exists():
        raise RuntimeError(f"Expected a symlink destination, found existing path: {destination}")
    else:
        destination.symlink_to(source, target_is_directory=True)

for sequence in sorted(EXPECTED_SEQUENCES):
    destination = SPRING_ROOT / "test" / sequence
    ensure_directory_link(LEFT_ROOT / "test" / sequence / "frame_left", destination / "frame_left")
    ensure_directory_link(RIGHT_ROOT / "test" / sequence / "frame_right", destination / "frame_right")
    ensure_directory_link(CAM_ROOT / "test" / sequence / "cam_data", destination / "cam_data")
    if not (destination / "cam_data" / "intrinsics.txt").is_file():
        raise FileNotFoundError(destination / "cam_data" / "intrinsics.txt")

sequence_frames = {}
expected_flow_files = 0
for sequence in sorted(EXPECTED_SEQUENCES):
    sequence_frames[sequence] = {}
    for side in ("left", "right"):
        frames = sorted((SPRING_ROOT / "test" / sequence / f"frame_{side}").glob(f"frame_{side}_*.png"))
        if len(frames) < 2:
            raise RuntimeError(f"Too few {side} frames for sequence {sequence}: {len(frames)}")
        numbers = [int(path.stem.rsplit("_", 1)[1]) for path in frames]
        if numbers != list(range(1, len(frames) + 1)):
            raise RuntimeError(f"Non-contiguous {side} frame numbering in sequence {sequence}")
        sequence_frames[sequence][side] = len(frames)
        expected_flow_files += 2 * (len(frames) - 1)
    if sequence_frames[sequence]["left"] != sequence_frames[sequence]["right"]:
        raise RuntimeError(f"Left/right frame mismatch in sequence {sequence}: {sequence_frames[sequence]}")

print("Canonical Spring root:", SPRING_ROOT)
print(json.dumps(sequence_frames, indent=2))
print("Expected FW/BW files across both cameras:", expected_flow_files)
"""),
    markdown(r"""
## 4. Install official MEMFOF and download its own checkpoint

This notebook no longer ports a PTLFlow copy. It installs the authors' current package and
uses `MEMFOF.from_pretrained`, exactly as documented by the official repository. The model
snapshot is revision-pinned and downloaded before the two GPU workers start.
"""),
    code(r"""
if not (MEMFOF_DIR / ".git").is_dir():
    subprocess.run([
        "git", "clone", "--branch", MEMFOF_SOURCE_BRANCH, "--single-branch",
        MEMFOF_REPO, str(MEMFOF_DIR),
    ], check=True)
subprocess.run(["git", "fetch", "origin", MEMFOF_SOURCE_REF], cwd=MEMFOF_DIR, check=True)
subprocess.run(["git", "checkout", "--detach", MEMFOF_SOURCE_REF], cwd=MEMFOF_DIR, check=True)
MEMFOF_SOURCE_COMMIT = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=MEMFOF_DIR, text=True
).strip()

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(MEMFOF_DIR)], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "h5py", "pillow", "tqdm"], check=True)

from huggingface_hub import snapshot_download

CHECKPOINT_DIR = Path(snapshot_download(
    repo_id=CHECKPOINT_REPO_ID,
    revision=CHECKPOINT_REVISION,
    allow_patterns=["config.json", "model.safetensors", "README.md"],
))
weights_file = CHECKPOINT_DIR / "model.safetensors"
if not weights_file.is_file() or weights_file.stat().st_size < 250_000_000:
    raise RuntimeError(f"MEMFOF weights are absent or incomplete: {weights_file}")

def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

CHECKPOINT_SHA256 = sha256(weights_file)
print("MEMFOF source commit:", MEMFOF_SOURCE_COMMIT)
print("Official MEMFOF snapshot:", CHECKPOINT_DIR)
print("model.safetensors:", weights_file.stat().st_size, "bytes")
print("model.safetensors SHA-256:", CHECKPOINT_SHA256)

if RAFT_CHECKPOINT.is_file():
    print("Ignored incompatible RAFT checkpoint:", RAFT_CHECKPOINT)
else:
    print("RAFT checkpoint is not attached (it is not needed for this MEMFOF run).")
assert "raft" not in str(CHECKPOINT_DIR).lower()
"""),
    markdown(r"""
## 5. Verify the model interface on CPU

The probe confirms that the downloaded files instantiate the real official architecture.
CUDA is intentionally untouched before the independent GPU subprocesses are launched.
"""),
    code(r"""
if str(MEMFOF_DIR) not in sys.path:
    sys.path.insert(0, str(MEMFOF_DIR))
from memfof import MEMFOF, AVAILABLE_MODELS

assert "MEMFOF-Tartan-T-TSKH" in AVAILABLE_MODELS
model_probe = MEMFOF.from_pretrained(str(CHECKPOINT_DIR), local_files_only=True).eval()
parameter_count = sum(parameter.numel() for parameter in model_probe.parameters())
assert 75_000_000 < parameter_count < 77_000_000, parameter_count
assert model_probe.corr_levels == 4 and model_probe.corr_radius == 4
print(f"Verified official MEMFOF: {parameter_count / 1e6:.2f}M parameters")
print("Three frames; output order: backward, forward; refinement iterations:", ITERATIONS)
del model_probe

# Verify the fresh subprocess import path used by the GPU workers. Keeping the worker inside
# the repository avoids /kaggle/working/memfof being mistaken for an empty namespace package.
import_test_env = os.environ.copy()
import_test_env["PYTHONPATH"] = str(MEMFOF_DIR) + (
    os.pathsep + import_test_env["PYTHONPATH"] if import_test_env.get("PYTHONPATH") else ""
)
subprocess.run(
    [sys.executable, "-c", "from memfof import MEMFOF; print('Worker import verified:', MEMFOF.__name__)"],
    cwd=MEMFOF_DIR, env=import_test_env, check=True,
)
"""),
    markdown(r"""
## 6. Create the two-GPU native-resolution inference worker

Each worker owns ten `(sequence, camera)` tasks and one GPU. For a center frame `i`, the
window is `[i−1, i, i+1]`; the first/last context index is clamped at sequence boundaries.
Only valid adjacent flows are written: `FW_i` for `i < N`, and `BW_i` for `i > 1`.
"""),
    code(r"""
INFERENCE_SCRIPT.write_text(r'''
import argparse
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
import torch
from tqdm.auto import tqdm

from memfof import MEMFOF


def read_frame(path):
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1)


def write_flo5(path, flow):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("flow", data=flow.astype(np.float32), compression="gzip", compression_opts=5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spring-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    model = MEMFOF.from_pretrained(str(args.checkpoint), local_files_only=True).eval().to(device)
    tasks = [
        (sequence, side)
        for sequence in sorted(path.name for path in (args.spring_root / "test").iterdir() if path.is_dir())
        for side in ("left", "right")
    ][args.shard_id::args.num_shards]

    with torch.inference_mode():
        for sequence, side in tqdm(tasks, desc=f"GPU shard {args.shard_id}"):
            frames = sorted((args.spring_root / "test" / sequence / f"frame_{side}").glob(f"frame_{side}_*.png"))
            fmap_cache = [None, None, None]
            for center in range(len(frames)):
                indices = [max(center - 1, 0), center, min(center + 1, len(frames) - 1)]
                images = torch.stack([read_frame(frames[index]) for index in indices], dim=0)
                images = images.unsqueeze(0).to(device, non_blocking=True)
                prediction = model(images, iters=args.iters, fmap_cache=fmap_cache)
                backward, forward = prediction["flow"][-1][0].float().cpu().numpy()

                fmap_cache = prediction["fmap_cache"]
                fmap_cache.pop(0)
                fmap_cache.append(None)
                frame_number = int(frames[center].stem.rsplit("_", 1)[1])
                if center > 0:
                    target = args.output_root / sequence / f"flow_BW_{side}" / f"flow_BW_{side}_{frame_number:04d}.flo5"
                    write_flo5(target, backward.transpose(1, 2, 0))
                if center < len(frames) - 1:
                    target = args.output_root / sequence / f"flow_FW_{side}" / f"flow_FW_{side}_{frame_number:04d}.flo5"
                    write_flo5(target, forward.transpose(1, 2, 0))


if __name__ == "__main__":
    main()
''')
print("Wrote inference worker:", INFERENCE_SCRIPT)
"""),
    markdown(r"""
## 7. Run MEMFOF on both T4 GPUs

The scoped output directory is cleared on rerun so a stale partial run cannot masquerade as
a complete submission. Both subprocesses must exit successfully before validation begins.
"""),
    code(r"""
if OUTPUT_ROOT.parent.exists():
    shutil.rmtree(OUTPUT_ROOT.parent)
OUTPUT_ROOT.mkdir(parents=True)

common = [
    sys.executable, "-u", str(INFERENCE_SCRIPT),
    "--spring-root", str(SPRING_ROOT),
    "--output-root", str(OUTPUT_ROOT),
    "--checkpoint", str(CHECKPOINT_DIR),
    "--iters", str(ITERATIONS),
    "--num-shards", str(NUM_GPUS),
]
processes = []
for gpu_id in range(NUM_GPUS):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(MEMFOF_DIR) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    command = common + ["--shard-id", str(gpu_id)]
    print("Launching:", " ".join(command), "with CUDA_VISIBLE_DEVICES=", gpu_id)
    processes.append(subprocess.Popen(command, env=env))

elapsed = time.monotonic() - SESSION_START
timeout_seconds = MAX_SESSION_HOURS * 3600 - PACKAGING_RESERVE_MINUTES * 60 - elapsed
deadline = time.monotonic() + timeout_seconds
while any(process.poll() is None for process in processes):
    if time.monotonic() >= deadline:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        raise TimeoutError("Inference reserve reached; workers were terminated before packaging time.")
    time.sleep(15)

return_codes = [process.returncode for process in processes]
if any(code != 0 for code in return_codes):
    raise RuntimeError(f"MEMFOF worker failure: return codes {return_codes}")
print("Both MEMFOF workers completed.")
"""),
    markdown(r"""
## 8. Validate every prediction before packaging

Checks cover exact path names, exact expected membership, native dimensions, dtype, and finite
values. This catches swapped directions, boundary off-by-one errors, and partial GPU failures.
"""),
    code(r"""
import h5py
import numpy as np

expected_paths = set()
for sequence in sorted(EXPECTED_SEQUENCES):
    count = sequence_frames[sequence]["left"]
    for side in ("left", "right"):
        expected_paths.update(
            Path(sequence) / f"flow_FW_{side}" / f"flow_FW_{side}_{index:04d}.flo5"
            for index in range(1, count)
        )
        expected_paths.update(
            Path(sequence) / f"flow_BW_{side}" / f"flow_BW_{side}_{index:04d}.flo5"
            for index in range(2, count + 1)
        )

files = sorted(OUTPUT_ROOT.rglob("*.flo5"))
actual_paths = {path.relative_to(OUTPUT_ROOT) for path in files}
missing = sorted(expected_paths - actual_paths, key=str)
unexpected = sorted(actual_paths - expected_paths, key=str)
if missing or unexpected:
    raise RuntimeError(
        f"Prediction membership mismatch: {len(missing)} missing, {len(unexpected)} unexpected; "
        f"examples missing={missing[:5]}, unexpected={unexpected[:5]}"
    )
if len(files) != expected_flow_files:
    raise RuntimeError(f"Expected {expected_flow_files} flow files; found {len(files)}")

name_re = re.compile(r"flow_(FW|BW)_(left|right)_\d{4}\.flo5$")
for path in files:
    if not name_re.fullmatch(path.name):
        raise RuntimeError(f"Invalid prediction filename: {path}")
    with h5py.File(path, "r") as handle:
        if "flow" not in handle or handle["flow"].shape != (1080, 1920, 2):
            raise RuntimeError(f"Invalid flow shape in {path}")
        if handle["flow"].dtype != np.float32:
            raise RuntimeError(f"Invalid flow dtype in {path}: {handle['flow'].dtype}")

for index in sorted({0, len(files) // 3, 2 * len(files) // 3, len(files) - 1}):
    with h5py.File(files[index], "r") as handle:
        if not np.isfinite(handle["flow"][:]).all():
            raise RuntimeError(f"NaN or Inf in {files[index]}")

size_gib = sum(path.stat().st_size for path in files) / 1024**3
print(f"Validated {len(files):,} native-resolution .flo5 files ({size_gib:.2f} GiB)")
print("Prediction root:", OUTPUT_ROOT)
"""),
    markdown(r"""
## 9. Package and record the artifact

The executable is copied out of read-only Kaggle input before changing its mode. The manifest
records both immutable checkpoint identity and the resolved official source commit.
"""),
    code(r"""
if not SUBSAMPLING_SOURCE.is_file():
    raise FileNotFoundError(f"Missing official flow_subsampling executable: {SUBSAMPLING_SOURCE}")
TOOL = WORK_ROOT / "flow_subsampling"
shutil.copy2(SUBSAMPLING_SOURCE, TOOL)
TOOL.chmod(0o755)

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
subprocess.run([str(TOOL), str(OUTPUT_ROOT)], cwd=ARTIFACT_DIR, check=True)
submission_files = sorted(ARTIFACT_DIR.glob("*.hdf5"), key=lambda path: path.stat().st_mtime)
if not submission_files:
    raise RuntimeError("flow_subsampling produced no HDF5 artifact.")
SUBMISSION_FILE = submission_files[-1]

manifest = {
    "team_id": TEAM_ID,
    "team_name": TEAM_NAME,
    "experiment": "03_memfof_tartan_t_tskh_zeroshot_native",
    "model": MODEL_NAME,
    "model_class": "official MEMFOF",
    "parameter_count": parameter_count,
    "checkpoint_repo_id": CHECKPOINT_REPO_ID,
    "checkpoint_revision": CHECKPOINT_REVISION,
    "checkpoint_file": "model.safetensors",
    "checkpoint_sha256": CHECKPOINT_SHA256,
    "ignored_incompatible_checkpoint": str(RAFT_CHECKPOINT),
    "spring_training": False,
    "fine_tuning": False,
    "test_time_augmentation": False,
    "iterations": ITERATIONS,
    "sequence_length": SEQUENCE_LENGTH,
    "directions": ["BW", "FW"],
    "resolution": "1920x1080_native",
    "gpus": NUM_GPUS,
    "memfof_source_repo": MEMFOF_REPO,
    "memfof_source_commit": MEMFOF_SOURCE_COMMIT,
    "prediction_files": len(files),
    "submission_file": SUBMISSION_FILE.name,
    "submission_bytes": SUBMISSION_FILE.stat().st_size,
    "submission_sha256": sha256(SUBMISSION_FILE),
}
manifest_path = ARTIFACT_DIR / "experiment_03_manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
print(manifest_path.read_text())
print("UPLOAD THIS FILE:", SUBMISSION_FILE)
print("Also save:", manifest_path)
"""),
    markdown(r"""
## Submission checklist

1. Enable **Internet** and select **GPU T4 ×2**.
2. Run `Save Version → Save & Run All`.
3. Download the generated `.hdf5` plus `experiment_03_manifest.json`.
4. Upload the `.hdf5` to the Spring optical-flow benchmark.

Declare: official **MEMFOF-Tartan-T-TSKH** checkpoint, no Spring training, no TTA, native
resolution. Do not describe the run as using `raft-sintel-fb44381e.ckpt`; that file is merely
attached and explicitly ignored by this notebook.
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {
            "accelerator": "nvidiaTeslaT4",
            "dataSources": [],
            "isInternetEnabled": True,
            "language": "python",
            "sourceType": "notebook",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

target = Path(__file__).with_name("exp03-memfof-zeroshot-baseline.ipynb")
target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(target)
