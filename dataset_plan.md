# Optical Flow Dataset Plan

This plan prioritizes the cheapest useful Spring downloads for Kaggle T4×2 sessions. Download each dataset once and reuse it across experiments.

## Stage 1: Clean baseline inference

Download only:

| ZIP file | Compressed size |
| --- | ---: |
| `test_frame_left.zip` | 2.8 GB |
| `test_frame_right.zip` | 2.8 GB |
| `test_cam_data.zip` | 3.7 KB |
| **Total** | **approximately 5.6 GB** |

This bundle supports the clean Spring test submission and the following inference variations without downloading more image data:

1. RAFT-Sintel, 32 iterations, native resolution (master baseline)
2. RAFT-Sintel, 12 iterations
3. RAFT-Sintel, 24 iterations
4. RAFT-Sintel, 48 iterations
5. RAFT-Sintel, 50% input resolution
6. RAFT-Sintel, 75% input resolution
7. RAFT-Sintel with warm start
8. RAFT-Things checkpoint
9. RAFT-KITTI checkpoint
10. RAFT-Small-Things
11. FP16 versus FP32
12. Horizontal-flip test-time augmentation

The alternative checkpoints add relatively small downloads compared with the image datasets.

Spring test ground truth is private. These runs can provide runtime, memory, qualitative results, and benchmark results after submission, but they cannot produce local EPE or WAUC. Because leaderboard submissions may be rate limited, submit only the master baseline and one or two selected alternatives.

## Stage 2: Cheapest locally measurable experiments

The smallest labeled Spring combination for optical-flow validation and fine-tuning is:

| ZIP file | Compressed size |
| --- | ---: |
| `train_frame_left.zip` | 12.4 GB |
| `train_flow_FW_left.zip` | 44.0 GB |
| **Training total** | **56.4 GB** |

Together with the clean test bundle, the cumulative compressed download is approximately **62.0 GB**.

This bundle supports:

- Zero-shot RAFT validation
- Spring fine-tuning
- Learning-rate and training-duration ablations
- Recurrent-iteration ablations
- Input-resolution ablations
- Maximum-flow loss ablations
- Crop-size ablations
- Synthetic corruption training and validation
- Corruption-component ablations

Use Spring sequence `0022` as the held-out clean validation sequence. Generate deterministic corrupted copies of the validation images while retaining their optical-flow labels. Generate training corruptions on the fly instead of downloading RobustSpring.

## Recommended low-download paper experiments

All experiments below reuse the same 56.4 GB labeled bundle:

| ID | Experiment |
| ---: | --- |
| 1 | Zero-shot RAFT-Sintel |
| 2 | Standard Spring fine-tuning |
| 3 | Spring fine-tuning with `max_flow=1000` |
| 4 | Fine-tuning with Gaussian-noise augmentation |
| 5 | Fine-tuning with Gaussian- and motion-blur augmentation |
| 6 | Fine-tuning with brightness and contrast augmentation |
| 7 | Fine-tuning with JPEG compression and pixelation |
| 8 | Balanced mixed-corruption training |
| 9 | Mixed corruption without noise (`-noise` ablation) |
| 10 | Mixed corruption without blur (`-blur` ablation) |
| 11 | Mixed corruption without photometric augmentation (`-photometric` ablation) |
| 12 | Best method with 12, 24, and 32 RAFT iterations |
| 13 | Best method at 0.50, 0.75, and native input resolution |

For photometric, noise, blur, JPEG, and pixelation transformations, the optical-flow labels remain usable when the transformation does not change image geometry. Do not apply elastic geometric transformations without transforming the flow labels consistently.

## Downloads deferred until necessary

Do not download these during the initial experiment phase:

- `train_frame_right.zip`
- `train_flow_FW_right.zip`
- `train_flow_BW_left.zip`
- `train_flow_BW_right.zip`
- `train_maps.zip`
- All `disp1_*.zip` and `disp2_*.zip` archives
- The 20 RobustSpring corruption archives

The disparity archives belong to Stereo Matching and Scene Flow, not the Optical Flow track.

## RobustSpring limitation

RobustSpring corruption archives contain test images but no public optical-flow ground truth. Downloading one or two corruption ZIPs permits pipeline tests and qualitative inspection, but not local accuracy measurement. A valid official robustness package requires predictions for the complete expected corruption set.

All 20 RobustSpring corruption ZIPs total approximately **101.8 GB compressed**. With the 5.6 GB clean test bundle, the complete robustness input is approximately **107.4 GB compressed**, before extraction. Defer this download until the clean baseline and local experiments work.

## Execution order

1. Download the 5.6 GB clean test bundle.
2. Run and package the RAFT-Sintel master baseline.
3. Run cheap inference variations on the same test images.
4. Download the 56.4 GB left-camera forward-flow training bundle.
5. Perform clean and synthetic-corruption validation using sequence `0022`.
6. Fine-tune and ablate the proposed corruption-robust method.
7. Download RobustSpring only when ready to generate the complete official robustness submission.

## Storage warning

All values above are compressed ZIP sizes reported in `details.txt`. Extracted datasets require substantially more storage. Extract archives before uploading them as Kaggle Datasets so extraction does not consume the limited 12-hour GPU session.
