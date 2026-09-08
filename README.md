# Flow State — RoCo-Spring 2026

Participation repository for **Flow State** in [RoCo-Spring: The Robust Correspondence Challenge](https://roco-spring.github.io/index.html), a NeurIPS 2026 challenge on robust dense correspondence under realistic distribution shifts.

RoCo-Spring evaluates both clean accuracy and robustness to realistic camera noise, adverse weather, blur, compression, and illumination changes. It uses the high-resolution [Spring](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-3376) benchmark and its 20-corruption [RobustSpring](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-5047) extension.

## Team registration

| Field | Details |
| --- | --- |
| Team ID | **RoCo-45** |
| Registered team name | **Flow State** |
| Primary contact | Syed Mohaiminul Hoque |
| Contact email | syedmuhaimintahsin@gmail.com |
| Affiliation | Center For Computational and Data Sciences Lab, Independent University Bangladesh |
| Registered | September 8, 2026 |
| Registration revision | 2 |
| Organizer record | Synchronized at revision 2 |
| Account status | Awaiting organizer verification |

The exact team ID and registered team name—**RoCo-45, Flow State**—must appear in the workshop paper abstract.

## Selected tracks

- Optical Flow — dense 2D motion between consecutive frames
- Stereo Matching — dense disparity from rectified stereo pairs
- Scene Flow — dense 3D motion from stereo image sequences
- Exploration Track — analysis of robustness, failure modes, method behavior, metrics, or evaluation design

## Competition schedule

| Date | Milestone |
| --- | --- |
| July 2, 2026 | Website launch |
| July 13, 2026 | Development leaderboard opened |
| **September 15, 2026** | **Workshop paper deadline (4–6 main pages)** |
| September 29, 2026 | Workshop paper author notification |
| **September 30, 2026** | **Final quantitative submission deadline** |
| **October 7, 2026** | **Camera-ready paper, code, and reproducibility package deadline** |
| October 15, 2026 | Final evaluation, reproducibility checks, and award shortlist |
| October 31, 2026 | Winners notified and workshop program finalized |
| December 11 or 12, 2026 | In-person NeurIPS Competition Track workshop |

All dates are tentative. Verify them on the [official challenge website](https://roco-spring.github.io/index.html) before submission.

## Participation workflow

- [x] Register the team
- [x] Receive team ID `RoCo-45`
- [ ] Receive organizer account verification
- [ ] Download the Spring and RobustSpring training splits
- [ ] Install the [official PTLFlow-based Starter Kit](https://github.com/hmorimitsu/roco-spring-devkit)
- [ ] Run baseline configurations and verify data paths and inference
- [ ] Train or adapt models and generate predictions for the selected tracks
- [ ] Validate prediction layout, names, and file formats locally
- [ ] Upload validated HDF5 submissions to the [Spring benchmark](https://spring-benchmark.org/)
- [ ] Submit the workshop paper through the [RoCo-Spring OpenReview venue](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FRoCo-Spring)
- [ ] Submit the camera-ready paper, code, and reproducibility package

## Quantitative submission requirements

Generate predictions on the Spring test split using the dataset's Python I/O utilities for `.flo5` and `.dsp5` files. Preserve the exact dataset sequence numbers, directory layout, and filenames.

### Standard evaluation layout

```text
<rootdir>/####/disp1_{left|right}/disp1_{left|right}_####.dsp5
<rootdir>/####/flow_{FW|BW}_{left|right}/flow_{FW|BW}_{left|right}_####.flo5
<rootdir>/####/disp2_{FW|BW}_{left|right}/disp2_{FW|BW}_{left|right}_####.dsp5
```

These paths correspond to Stereo Matching, Optical Flow, and the additional disparity-over-time output required for Scene Flow, respectively.

### Optional robustness evaluation layout

```text
<rootdir>/<corruption>/test/####/disp1_{left|right}/disp1_{left|right}_####.dsp5
<rootdir>/<corruption>/test/####/flow_{FW|BW}_{left|right}/flow_{FW|BW}_{left|right}_####.flo5
<rootdir>/<corruption>/test/####/disp2_{FW|BW}_{left|right}/disp2_{FW|BW}_{left|right}_####.dsp5
```

Supported top-level folders are `clean` plus the 20 corruptions: `brightness`, `contrast`, `defocus_blur`, `elastic_transform`, `fog`, `frost`, `gaussian_blur`, `gaussian_noise`, `glass_blur`, `impulse_noise`, `jpeg_compression`, `motion_blur`, `pixelate`, `rain`, `saturate`, `shot_noise`, `snow`, `spatter`, `speckle_noise`, and `zoom_blur`.

Run the appropriate official subsampling executable from the submission root to produce an upload-ready `.hdf5` file:

```bash
./disp1_subsampling <rootdir>       # Stereo Matching
./flow_subsampling <rootdir>        # Optical Flow
./disp2_subsampling <rootdir>       # Scene Flow disparity over time
```

Use the corresponding `*_robust_subsampling` executables for robustness submissions. A Scene Flow submission requires all three generated files. Benchmark evaluation takes approximately 1–2 hours; results are private by default and may later be made public anonymously or with the team and method name.

## Workshop paper

- Use the official NeurIPS 2026 format with the `sglblindworkshop` option.
- Submit a single-blind paper with 4–6 main-content pages; references and an optional appendix may be outside that limit.
- Include the exact identifier and name **RoCo-45, Flow State** in the abstract.
- Submit the reproducibility package with the camera-ready paper.

## Official resources

- [Challenge overview](https://roco-spring.github.io/index.html)
- [Participation instructions](https://roco-spring.github.io/participate.html)
- [OpenReview submission venue](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FRoCo-Spring)
- [Starter Kit](https://github.com/hmorimitsu/roco-spring-devkit)
- [Spring benchmark](https://spring-benchmark.org/)
- [Spring dataset](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-3376)
- [RobustSpring dataset](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-5047)
- Challenge support: roco-spring-org@googlegroups.com

Information last checked against the official challenge and participation pages on **September 8, 2026**.
