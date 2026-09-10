# RoCo-45 Optical-Flow Improvement Plan

## Current position

RoCo-45 scored **6.766% total 1px**, placing 45th in the supplied Spring leaderboard. This is effectively an exact reproduction of the organizer's RAFT baseline, which scored 6.765%. The submission pipeline is therefore working; the main limitation is the RAFT-Sintel baseline rather than packaging or inference correctness.

| Target | 1px cutoff | Reduction needed from 6.766 |
| --- | ---: | ---: |
| Top 40 | 6.323 | 6.5% |
| Top 30 | 4.565 | 32.5% |
| Top 20 | 3.716 | 45.1% |
| Top 10 | 3.433 | 49.3% |
| Top 5 | 3.265 | 51.7% |

The largest error slices are:

| Slice | RoCo-45 1px |
| --- | ---: |
| High detail | 64.231 |
| Large motion, 40+ px | 41.582 |
| Unmatched | 38.165 |
| Sky | 29.292 |
| Non-rigid | 27.987 |
| Rigid | 3.961 |

High-detail and unmatched pixels have large error but relatively small benchmark weight. Improving rigid motion, large displacements, and non-rigid motion should have more effect on total 1px and EPE.

## Next notebook

The next notebook should be:

> **Experiment 02 — Spring Local Validation and RAFT Fine-Tuning Baseline**

Its purpose is to build a trustworthy local validation loop before trying additional leaderboard submissions. It should:

1. Load the labeled Spring left-camera forward-flow training bundle.
2. Hold out sequence `0022` for validation.
3. Reproduce zero-shot RAFT-Sintel metrics locally.
4. Fine-tune RAFT on the remaining Spring sequences.
5. Compare `max_flow=400` and `max_flow=1000`.
6. Report total 1px, EPE, Fl, WAUC, and flow-magnitude buckets.
7. Save checkpoints, CSV results, configuration, runtime, and validation predictions.

This notebook is an infrastructure and training control experiment. It is not expected to produce the final competitive model, but every later architecture and corruption experiment should reuse its split, metrics, logging, and artifact conventions.

## Dataset required for Experiment 02

Download and reuse:

| ZIP file | Compressed size |
| --- | ---: |
| `train_frame_left.zip` | 12.4 GB |
| `train_flow_FW_left.zip` | 44.0 GB |
| **Total** | **56.4 GB** |

Together with the existing 5.6 GB clean test bundle, the cumulative compressed download is approximately 62.0 GB.

### Required dataset configuration correction

The bundle above contains only left-camera frames and forward-flow labels. The existing generic configuration uses `spring-train` and `spring-val`, which default to both camera sides. Experiment 02 must instead use:

```yaml
data:
  train_dataset: spring-train-left
  val_dataset: spring-val-left
```

Do not request `timerev`, `timerevonly`, or `back` unless the corresponding backward-flow archive has also been downloaded.

## Phase 1 — RAFT fine-tuning controls

Run the following locally and evaluate every checkpoint on sequence `0022`:

| ID | Initialization | Crop | Training `max_flow` | Train iterations | Evaluation iterations |
| --- | --- | ---: | ---: | ---: | ---: |
| R0 | RAFT-Sintel, no training | Native validation | 400 | n/a | 32 |
| R1 | RAFT-Sintel | 540×960 | 400 | 12 | 32 |
| R2 | RAFT-Sintel | 540×960 | 1,000 | 12 | 32 |
| R3 | Best R1/R2 | 720×1280 if memory permits | Best | 12 | 32 |

Use validation 1px as the primary selection metric and EPE as the secondary metric. The `max_flow=1000` experiment is important because Spring includes large displacements and RAFT's sequence loss excludes ground-truth pixels above its configured maximum flow.

For the winning fine-tuned checkpoint, compare 12, 24, and 32 evaluation iterations locally. Do not submit each iteration setting to the leaderboard.

## Phase 2 — Move to a modern backbone

Fine-tuned RAFT is a control, not the intended final architecture. Evaluate stronger models in this order:

1. **MEMFOF**
2. **SEA-RAFT-M**
3. **DPFlow**
4. **WAFT-DAv2**

The pinned local challenge devkit contains only RAFT. These models must be integrated from their official repositories or a current PTLFlow release, then adapted to write the exact Spring `.flo5` directory structure used by the validated baseline notebook.

### MEMFOF experiments

MEMFOF is the first choice for native 1080p inference because it is memory-efficient and uses three-frame temporal context.

| ID | Model | Purpose |
| --- | --- | --- |
| M1 | MEMFOF-Tartan-T-TSKH | Zero-shot architecture comparison |
| M2 | Official Spring checkpoint | Pipeline verification and reference ceiling |
| M3 | M1 fine-tuned on Spring | Team-controlled Spring adaptation |

Native-resolution MEMFOF training is unlikely to fit on a 15 GB T4. Use crop training and gradient accumulation; keep native resolution for inference.

### SEA-RAFT experiments

SEA-RAFT-M is the lower-risk two-frame alternative and is particularly relevant to RoCo-45's rigid-motion errors.

| ID | Model | Purpose |
| --- | --- | --- |
| S1 | General/zero-shot SEA-RAFT-M checkpoint | Architecture comparison |
| S2 | Official Spring-M checkpoint | Pipeline verification |
| S3 | S1 fine-tuned on Spring | Team-controlled adaptation |
| S4 | S3 with mixed-corruption training | Robust final candidate |

### DPFlow experiment

DPFlow is a useful fallback because the supplied leaderboard shows a clear Spring fine-tuning gain:

- DPFlow without fine-tuning: 4.277% 1px.
- DPFlow with fine-tuning: 3.442% 1px.
- Absolute improvement: 0.835 percentage points.

This places successful DPFlow fine-tuning near the current top-10 boundary.

### WAFT experiment

WAFT-DAv2-a2 scores 3.298% on the supplied leaderboard, but it has a more complex environment and backbone setup. Attempt it after the local validation, output writing, and fine-tuning pipelines are stable.

## Phase 3 — Fine-tune the winning modern model

Run this compact experiment matrix using the best practical modern backbone:

| ID | Change | Purpose |
| --- | --- | --- |
| F0 | Pretrained checkpoint without Spring training | Local baseline |
| F1 | Standard Spring fine-tuning | Domain adaptation |
| F2 | `max_flow`: 400 → 1,000 | Retain large-flow supervision |
| F3 | Larger crop or broader scale augmentation | Increase motion range and context |
| F4 | Flow-magnitude-balanced crop sampling | Prevent small motions dominating training |
| F5 | 12/24/32 inference iterations | Accuracy/runtime selection |
| F6 | Horizontal-flip TTA | Final small-gain experiment |

Only submit a model to the clean leaderboard after it clearly improves held-out validation and passes the same filename, count, shape, finite-value, and packaging checks as Experiment 01.

## Phase 4 — Corruption-consistent fine-tuning

Once clean validation reaches the approximate 3.3–3.8% 1px range, add synthetic corruptions while retaining a substantial clean-data fraction:

- 50% clean samples.
- 50% corrupted samples.
- Noise: Gaussian, shot-like, impulse, and speckle approximations.
- Blur: Gaussian, defocus, and motion blur.
- Photometric: brightness, contrast, saturation.
- Quality: JPEG compression and pixelation.

Keep all transformations label-preserving. Do not apply elastic or other geometric transformations unless the optical-flow labels are transformed consistently.

Track clean accuracy and average corruption degradation separately. A useful selection objective is

$$
J=\operatorname{1px}_{\mathrm{clean}}+
\lambda\frac{1}{K}\sum_{k=1}^{K}
\left(\operatorname{1px}_{k}-\operatorname{1px}_{\mathrm{clean}}\right),
$$

where $K$ is the number of validation corruptions. Test $\lambda\in\{0.25,0.5\}$.

A clean-to-corrupted consistency term can be added as

$$
\mathcal L=
\mathcal L_{\mathrm{flow}}(\widehat F_c,F^*)+
\beta\left\|\widehat F_c-
\operatorname{stopgrad}(\widehat F_{\mathrm{clean}})\right\|_1.
$$

This provides a stronger RoCo-Spring contribution than ordinary iteration or checkpoint tuning.

## Experiments to deprioritize

Do not spend scarce leaderboard submissions on these before local validation:

- RAFT with 12 versus 24 versus 48 iterations.
- RAFT-Things or RAFT-KITTI checkpoints.
- RAFT-Small.
- 50% input resolution.
- FP16 versus FP32.

These remain useful runtime or ablation controls, but they are unlikely to provide the 45–50% improvement required for the current top 20 or top 10. Reduced input resolution is especially likely to harm motion boundaries and large displacement.

Warm start must also be treated carefully. Before using it, verify that recurrent state resets before the first pair of every sequence and that forward and backward directions receive independent, temporally correct states.

## Submission policy

Use local sequence-`0022` validation to select hyperparameters. Reserve clean leaderboard submissions for:

1. The best zero-shot modern architecture.
2. The best clean Spring-fine-tuned model.
3. The final clean or corruption-aware model.

Do not download the complete 101.8 GB RobustSpring corruption set until the clean model and local synthetic-corruption experiments are complete. A valid official robustness submission requires predictions for the complete expected corruption set.

## Recommended execution order

1. Download the 56.4 GB labeled left-forward Spring bundle.
2. Build Experiment 02 with deterministic sequence-`0022` validation.
3. Run R0, R1, and R2; run R3 only if memory and time permit.
4. Integrate and validate MEMFOF zero-shot at native resolution.
5. Integrate SEA-RAFT-M as the two-frame comparison.
6. Select the better model using local 1px, EPE, memory, and runtime.
7. Fine-tune the selected model on Spring.
8. Run the `max_flow`, crop-size, and magnitude-sampling ablations.
9. Add balanced mixed-corruption consistency training.
10. Apply flip TTA only to the final candidate.
11. Retrain the selected configuration with all available clean training sequences.
12. Generate and validate the clean benchmark submission.
13. Download RobustSpring and generate the complete robustness submission.

## Expected milestones

- **Above top 40:** achievable with a modest improvement over RAFT.
- **Top 20:** realistically requires a modern pretrained architecture.
- **Top 10:** likely requires a strong modern model plus successful Spring fine-tuning.
- **Top 5:** likely requires a top architecture plus a meaningful adaptation, temporal fusion, uncertainty strategy, or ensemble improvement.

