"""Generate separate Kaggle notebooks for RAFT training and inference."""

import json
from pathlib import Path


ROOT = Path(__file__).parent
combined_path = ROOT / "raft-spring-finetune-inference/raft-spring-finetune-inference.ipynb"
combined = json.loads(combined_path.read_text())
source_cells = combined["cells"]


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(True),
    }


def source(index):
    return "".join(source_cells[index]["source"])


def write_notebook(relative_path, cells):
    target = ROOT / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    target.write_text(json.dumps(notebook, indent=1) + "\n")
    print(target)


training_config = '''from pathlib import Path
import importlib, json, os, platform, shutil, subprocess, sys

RUN_TRAINING = True
SEED = 3407
TRAIN_EPOCHS = 1
TRAIN_BATCH_SIZE = 1
ACCUMULATE_GRAD_BATCHES = 4
TRAIN_ITERS = 12
TRAIN_LR = 1.0e-5
TRAIN_WEIGHT_DECAY = 1.0e-5
TRAIN_CROP_SIZE = [540, 960]
TRAIN_PRECISION = "16-mixed"
DEVKIT_REF = "90ae81a9324c6806dc3c2482aab84a2744215bd9"

KAGGLE_INPUT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working")
DEVKIT_DIR = WORK_ROOT / "roco-spring-devkit"
TRAIN_SPRING_ROOT = KAGGLE_INPUT / "datasets/sakhawatdhrubo2/spring-train-left-10seq/spring_subset/spring"
TRAIN_MANIFEST = KAGGLE_INPUT / "datasets/sakhawatdhrubo2/spring-train-left-10seq/spring_subset/manifest.json"
PRETRAINED_CHECKPOINT = KAGGLE_INPUT / "datasets/strikingratio/ckpoint/raft-sintel-fb44381e.ckpt"
EXPECTED_TRAIN_SEQUENCES = {"0011", "0022", "0025", "0026", "0027", "0030", "0032", "0036", "0041", "0045"}
TRAIN_LOG_DIR = WORK_ROOT / "raft_spring_train_logs"
FINETUNED_CHECKPOINT = WORK_ROOT / "raft-spring-left-finetuned.ckpt"

assert KAGGLE_INPUT.is_dir(), "Run this notebook in Kaggle."
WORK_ROOT.mkdir(parents=True, exist_ok=True)
print(json.dumps({
    "train_root": str(TRAIN_SPRING_ROOT),
    "epochs": TRAIN_EPOCHS,
    "iterations": TRAIN_ITERS,
    "output_checkpoint": str(FINETUNED_CHECKPOINT),
}, indent=2))
'''
training_run = source(7).split('else:\n    print("Training skipped.")')[0]
training_run += '\nprint("Training output checkpoint:", FINETUNED_CHECKPOINT)\n'
training_cells = [
    markdown("""# Session 1 — Fine-tune RAFT-Sintel on Spring

This notebook only trains. It starts from the supplied RAFT-Sintel checkpoint, fine-tunes on the left-camera forward-flow subset, validates on held-out sequence `0022`, and writes one portable checkpoint:

`/kaggle/working/raft-spring-left-finetuned.ckpt`

After the run completes, save a Kaggle Notebook Version with outputs and create a Kaggle Dataset containing that checkpoint. Attach that output dataset to the separate inference notebook.
"""),
    code(training_config),
    source_cells[2],
    source_cells[3],
    source_cells[4],
    source_cells[5],
    source_cells[6],
    code(training_run),
    markdown("""## Training handoff

Download or publish `/kaggle/working/raft-spring-left-finetuned.ckpt` as a private Kaggle Dataset. The inference notebook searches attached inputs for this exact filename. Keep the training logs if you want to compare validation metrics across longer runs.

Start with one epoch. Increase `TRAIN_EPOCHS` only after this run succeeds and only while the held-out `0022` metric improves.
"""),
]


inference_config = '''from pathlib import Path
import hashlib, importlib, json, os, platform, re, shutil, subprocess, sys, time

RUN_TRAINING = False
RUN_INFERENCE = True
RUN_PACKAGING = True
MODEL = "raft"
INFERENCE_ITERS = 32
INFERENCE_CORR_MODE = "triton"
INFERENCE_GPUS = 2
MAX_FORWARD_SIDE = None
DEVKIT_REF = "90ae81a9324c6806dc3c2482aab84a2744215bd9"
# Keep these aligned with the checkpoint-producing training notebook.
TRAINING_EPOCHS_METADATA = 1
TRAINING_ITERS_METADATA = 12

KAGGLE_INPUT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working")
SCRATCH_ROOT = Path("/kaggle/temp")
DEVKIT_DIR = WORK_ROOT / "roco-spring-devkit"
TEST_LEFT_ROOT = KAGGLE_INPUT / "datasets/strikingratio/test-frame-left-right/test_frame_left/spring"
TEST_RIGHT_ROOT = KAGGLE_INPUT / "datasets/strikingratio/test-frame-left-right/test_frame_right/spring"
PRETRAINED_CHECKPOINT = KAGGLE_INPUT / "datasets/strikingratio/ckpoint/raft-sintel-fb44381e.ckpt"
SUBSAMPLING_INPUT = KAGGLE_INPUT / "datasets/strikingratio/flow-subsampling/flow_subsampling"
EXPECTED_TRAIN_SEQUENCES = {"0011", "0022", "0025", "0026", "0027", "0030", "0032", "0036", "0041", "0045"}

# Usually leave this as None for automatic discovery by filename.
FINETUNED_CHECKPOINT_OVERRIDE = None

SPRING_TEST_ROOT = SCRATCH_ROOT / "spring_test_merged"
OUTPUT_BASE = SCRATCH_ROOT / "raft_spring_finetuned_predictions"
ARTIFACT_DIR = WORK_ROOT / "raft_spring_finetuned_artifacts"
SESSION_START = time.monotonic()
MAX_SESSION_HOURS = 12.0
PACKAGING_RESERVE_MINUTES = 30

assert KAGGLE_INPUT.is_dir(), "Run this notebook in Kaggle."
for directory in (WORK_ROOT, SCRATCH_ROOT):
    directory.mkdir(parents=True, exist_ok=True)
'''
inference_preflight = r'''gpu = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
    text=True, capture_output=True, check=True,
)
print(gpu.stdout)
gpu_count = len([line for line in gpu.stdout.splitlines() if line.strip()])
if gpu_count < 1:
    raise RuntimeError("Enable a Kaggle GPU accelerator and restart the session.")

# You may set FINETUNED_CHECKPOINT_OVERRIDE in the configuration cell. Otherwise,
# search only shallow input metadata and avoid descending into image folders.
if FINETUNED_CHECKPOINT_OVERRIDE:
    INFERENCE_CHECKPOINT = Path(FINETUNED_CHECKPOINT_OVERRIDE)
else:
    checkpoint_matches = []
    for current, dirs, files in os.walk(KAGGLE_INPUT):
        current = Path(current)
        try:
            depth = len(current.relative_to(KAGGLE_INPUT).parts)
        except ValueError:
            continue
        dirs[:] = [d for d in dirs if d not in {"frame_left", "frame_right", "flow_FW_left"}]
        if "raft-spring-left-finetuned.ckpt" in files:
            checkpoint_matches.append(current / "raft-spring-left-finetuned.ckpt")
        if depth >= 7:
            dirs[:] = []
    if len(checkpoint_matches) != 1:
        raise RuntimeError(
            "Attach exactly one training-output dataset containing "
            f"raft-spring-left-finetuned.ckpt; found {checkpoint_matches}"
        )
    INFERENCE_CHECKPOINT = checkpoint_matches[0]

if not INFERENCE_CHECKPOINT.is_file() or INFERENCE_CHECKPOINT.stat().st_size < 1_000_000:
    raise RuntimeError(f"Fine-tuned checkpoint is missing or incomplete: {INFERENCE_CHECKPOINT}")
print("Fine-tuned checkpoint:", INFERENCE_CHECKPOINT)
print("Python:", sys.version.split()[0], "OS:", platform.platform())
'''

inference_install = source(5).replace(
    'probe = RAFT(iters=TRAIN_ITERS, corr_mode="allpairs", predict_all_directions=False)',
    'probe = RAFT(iters=INFERENCE_ITERS, corr_mode=INFERENCE_CORR_MODE, predict_all_directions=True)',
)
inference_package = source(14).replace(
    '"training_epochs": TRAIN_EPOCHS', '"training_epochs": TRAINING_EPOCHS_METADATA'
).replace(
    '"training_iterations": TRAIN_ITERS', '"training_iterations": TRAINING_ITERS_METADATA'
)

inference_cells = [
    markdown("""# Session 2 — Infer with the fine-tuned Spring RAFT checkpoint

This notebook does not train. Attach the private Kaggle Dataset produced by Session 1; it must contain `raft-spring-left-finetuned.ckpt`. The notebook discovers that file, runs native-resolution inference on both cameras and temporal directions, validates every `.flo5`, and creates the official Spring benchmark HDF5 artifact.

Recommended accelerator: **GPU T4 x2**. The separately attached left/right test-frame datasets and `flow_subsampling` executable are also required.
"""),
    code(inference_config),
    markdown("""## 1. Locate the checkpoint exported by Session 1

If automatic discovery is ambiguous, set `FINETUNED_CHECKPOINT_OVERRIDE` in the configuration cell to its full attached-input path.
"""),
    code(inference_preflight),
    source_cells[4],
    code(inference_install),
    source_cells[8],
    source_cells[9],
    source_cells[10],
    source_cells[11],
    source_cells[12],
    source_cells[13],
    code(inference_package),
    markdown("""## Output

Download the generated `.hdf5` and `raft_spring_finetuned_manifest.json` from `/kaggle/working/raft_spring_finetuned_artifacts`. Upload the HDF5 file to the Spring optical-flow benchmark.
"""),
]

write_notebook(
    "raft-spring-finetune-training/raft-spring-finetune-training.ipynb",
    training_cells,
)
write_notebook(
    "raft-spring-finetuned-inference/raft-spring-finetuned-inference.ipynb",
    inference_cells,
)
