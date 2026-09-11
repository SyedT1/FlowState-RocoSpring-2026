# How Modern Optical-Flow Models Are Trained for Spring and RobustSpring

## Executive answer

The standard way to train models for this task is **supervised, staged, full-parameter fine-tuning**, not QLoRA. A model is first trained on large synthetic optical-flow datasets, commonly FlyingChairs, FlyingThings3D, and sometimes TartanAir. It is then fine-tuned on a mixture of established benchmarks such as Sintel, KITTI, and HD1K, and finally fine-tuned on Spring at Spring's much higher resolution and motion range. Each training example consists of two or more video frames plus a dense two-dimensional flow vector for every valid pixel.

The strongest Spring-oriented recipes currently fall into three families:

1. **Full end-to-end supervised training:** RAFT, SEA-RAFT, DPFlow, and MEMFOF update essentially the whole flow network with dense ground-truth flow.
2. **Pretrained frozen visual backbone plus a trainable flow head:** WAFT freezes all or most of a pretrained image/depth/foundation backbone and trains the DPT-style flow head and side network. This is parameter-efficient adaptation, but it is not LoRA.
3. **Robustness adaptation:** clean/degraded paired training, augmentation, teacher–student consistency, pseudo-labeling, or feature distillation can be added to ordinary supervised flow training. This is especially relevant to RobustSpring, but its test corruptions must not be used as a tuning set.

QLoRA was designed for very large language models: it keeps a large pretrained model quantized to 4 bits and learns small low-rank adapters. None of the official training recipes reviewed for RAFT, SEA-RAFT, DPFlow, MEMFOF, or WAFT uses QLoRA. LoRA could be an experimental engineering choice for a transformer-based flow backbone, but it is not the established recipe and is unlikely to be the best first experiment for this competition.[^lora][^qlora]

## 1. What exactly is learned?

For two frames, \(I_1\) and \(I_2\), the network predicts a dense flow field \(f(x,y)=(u,v)\). The supervised label tells the model where each valid pixel in the first frame moved in the second. Recurrent models produce several increasingly refined flow predictions, \(f_1,\ldots,f_N\), and normally supervise all of them:

\[
L_{sequence}=\sum_{i=1}^{N}\gamma^{N-i}\,L(f_i,f^*),
\]

where \(f^*\) is the ground-truth flow and invalid or excessively large vectors are masked. Original RAFT uses an L1 endpoint loss. Newer SEA-RAFT, DPFlow, MEMFOF, and WAFT use variants of a probabilistic mixture-of-Laplace loss, allowing the network to express uncertainty or ambiguity around occlusions and difficult pixels.[^raft-paper][^sea-paper][^dpflow-paper][^memfof-paper][^waft-paper]

Spring is unusually demanding because its images are 1920×1080 while its ground truth is supplied at 3840×2160, and it contains detailed regions and very large motions. The benchmark reports a strict 1-pixel outlier rate as its main accuracy measure, together with EPE and region-specific breakdowns.[^spring-paper] The consequence is important: training only on small crops or aggressively downsampled images can teach the basic problem, but high-resolution fine-tuning matters for the last part of performance.

## 2. The training taxonomy

| Training type | What changes | Labels required? | Where it appears here | Recommendation |
|---|---|---:|---|---|
| Supervised training from scratch | All model weights | Dense flow | Original RAFT synthetic stages | Too expensive for this project |
| Supervised checkpoint fine-tuning | All or nearly all weights | Dense flow | RAFT, SEA-RAFT, DPFlow, MEMFOF Spring stages | Standard and recommended |
| Frozen-backbone/side-tuning | Flow head and side layers; backbone frozen | Dense flow | WAFT a1/a2 | Promising when memory is limited |
| Self-supervised training | All or selected weights | No dense labels; uses warping/consistency | SMURF, FlowFormer++ pretraining | Useful with substantial unlabeled video |
| Semi-supervised pseudo-labeling | Student model; often EMA teacher | Some labels plus unlabeled frames | DistractFlow, OCAI | Optional extension, not the first baseline |
| Domain adaptation/distillation | Selected or full student weights | Often clean/degraded pairs; sometimes flow labels | FlowAnyTime and related work | Highly relevant to corruptions, but more complex |
| LoRA/QLoRA | Small low-rank adapters; base frozen, possibly quantized | Depends on task | Common in LLMs, absent from reviewed official flow recipes | Do not start here |

Two terms are often confused:

- **Mixed-precision training** stores or computes many activations in FP16/BF16 while maintaining stable optimizer state. It is a normal memory and speed optimization for optical flow.
- **QLoRA** freezes a 4-bit-quantized base model and trains low-rank adapter matrices. It is a particular parameter-efficient fine-tuning algorithm, not another name for mixed precision.[^qlora]

## 3. What each important model actually does

### RAFT: the most useful control experiment

RAFT constructs an all-pairs correlation volume and iteratively updates the flow with a recurrent unit. Its original official recipe trains all modules, initially from random weights, and then performs staged supervised transfer:[^raft-paper][^raft-train]

| Stage | Steps | Batch | Crop | Learning rate | Weight decay |
|---|---:|---:|---|---:|---:|
| FlyingChairs | 100k | 10 | 368×496 | 4e-4 | 1e-4 |
| FlyingThings3D | 100k | 6 | 400×720 | 1.25e-4 | 1e-4 |
| Sintel/KITTI/HD1K/Things mixture | 100k | 6 | 368×768 | 1.25e-4 | 1e-5 |
| KITTI specialization | 50k | 6 | 288×960 | 1e-4 | 1e-5 |

The optimizer is AdamW, the learning-rate policy is OneCycle, gradients are clipped, and ordinary training uses 12 recurrent updates. More updates can be used at inference. Official augmentation includes random scaling/stretching, cropping, horizontal/vertical flips where appropriate, color jitter, asymmetric color augmentation, occlusion erasing, and sparse-flow resizing rules.[^raft-data]

For this project, there is no reason to reproduce RAFT's pretraining. Load a strong Sintel/Things checkpoint and fine-tune it on Spring. That is ordinary transfer learning: all weights remain trainable unless an explicit ablation freezes some layers.

### SEA-RAFT: full supervised training with a better output distribution

SEA-RAFT keeps the recurrent-flow idea but directly regresses an initial flow estimate, needs only four update iterations in its main recipe, and replaces plain L1 with a mixture-of-Laplace likelihood. It uses an ImageNet-pretrained ResNet-18 or ResNet-34 backbone.[^sea-paper]

Its published sequence is TartanAir (300k steps), Chairs (100k), Things (120k), a Things/Sintel/KITTI/HD1K mixture (300k), and then target-specific fine-tuning. The official Spring configuration uses 120k iterations, a 540×960 crop, effective batch 32 across eight L40 GPUs, AdamW, learning rate 4e-4, weight decay 1e-5, four recurrent iterations, sequence discount \(\gamma=0.85\), gradient clipping, and full supervised Spring labels.[^sea-train][^sea-spring-config]

This is not a small-adapter recipe. The whole flow model is fine-tuned. Its main lesson for a smaller setup is to start from the released checkpoint and reproduce the crop, loss, and validation protocol with a much smaller per-GPU batch and gradient accumulation.

### DPFlow: a strong convolutional model with an explicit Spring fine-tune

DPFlow uses a dual-pyramid, adaptive-resolution convolutional architecture rather than a large pretrained vision transformer. It is trained in five supervised stages. The paper reports Chairs for 100k steps, Things for 1M, mixed Things/HD1K/Sintel/KITTI training for 120k, and then target-specific stages. Its Spring stage is 120k iterations/100 epochs with 540×960 crops, AdamW, learning rate 1e-4, weight decay 1e-5, and an effective batch of 16 through two GPUs and gradient accumulation.[^dpflow-paper][^dpflow-config]

DPFlow's published cost is a warning: the Spring stage alone is reported at roughly 360 RTX-3090 GPU-hours. The public benchmark shows why it exists: the listed Spring 1-pixel error improves from 4.277 without Spring fine-tuning to 3.442 with it—an absolute improvement of 0.835, about 19.5% relative.[^spring-leaderboard] For a limited Kaggle session, a shortened continuation from the authors' checkpoint is realistic; reproducing the official five-stage schedule is not.

### MEMFOF: high-resolution, multi-frame supervised training

MEMFOF uses multiple frames and was designed to make high-resolution multi-frame flow training feasible. It skips FlyingChairs because Chairs supplies only frame pairs, whereas the model benefits from three-frame context and joint forward/backward reasoning. Its stages use TartanAir, Things, and the TSKH mixture, followed by task specialization. Earlier non-Spring stages upsample images and flow by 2× to expose the model to Spring-like resolutions and displacement ranges.[^memfof-paper]

The published Spring stage starts from the TSKH checkpoint and uses native 1080×1920 crops, eight iterations, batch 32, learning rate 4.8e-5, 60k steps, mixture-of-Laplace sequence loss with \(\gamma=0.85\), AdamW, OneCycle, and automatic mixed precision. It was trained as part of a 32-A100 setup; reported per-device training memory for native Spring is about 28.5 GB.[^memfof-paper][^memfof-code]

That official native-resolution training does not fit a 15-GB T4. Gradient accumulation can reproduce a large effective batch, but it cannot make a single 28.5-GB example fit. MEMFOF remains valuable as an inference checkpoint or as a research model trained with smaller crops, fewer frames/iterations, checkpointing, or selective freezing—but those are deviations from its published Spring recipe.

### WAFT: the closest thing here to parameter-efficient adaptation

WAFT integrates pretrained visual features into an optical-flow system. Its variants use ImageNet-pretrained Twins, depth-pretrained Depth Anything V2, or self-supervised DINOv3 features. WAFT-a1 freezes the Depth Anything V2 components and learns a refinement path. WAFT-a2 freezes the primary transformer/CNN backbone while allowing a DPT head and a small ResNet side path to train. It still uses a supervised mixture-of-Laplace flow loss.[^waft-paper]

Its staged schedule is TartanAir (300k), Chairs (50k), Things (200k), mixed/Sintel training (200k), and Spring (200k, nominal batch 32, learning rate 1e-4). A DAv2-a1 variant also receives native-1080p training at batch 8. This is feature-backbone freezing and side-tuning—not low-rank adaptation and not QLoRA.

WAFT is especially relevant to constrained hardware. The paper's batch-one memory table reports roughly 7.0–9.2 GB for the Twins-a2 version as feature resolution increases from one-eighth to one-half, versus 14.1 GB to out-of-memory for SEA-RAFT in the same comparison.[^waft-paper] Exact training memory will depend on crop, backbone, framework, and optimizer, but WAFT is a more plausible modern fine-tuning candidate on a 15-GB device than native-resolution MEMFOF.

### FlowFormer++: self-supervised pretraining is possible, but it is a separate stage

FlowFormer is a transformer over a tokenized cost volume and uses an ImageNet-pretrained Twins image encoder before supervised Chairs/Things training.[^flowformer] FlowFormer++ adds masked cost-volume autoencoding: it masks cost-volume patches and learns to reconstruct them before supervised fine-tuning. The image encoder is frozen during this self-supervised pretraining to prevent collapse; the resulting network is subsequently fully supervised on labeled optical-flow data.[^flowformerpp]

This is useful evidence that unlabeled pretraining can help optical flow, but it does not eliminate supervised fine-tuning. It is also a much larger project than starting from a released Spring-capable checkpoint.

## 4. What the Spring results imply

The live public leaderboard changes, so values must be treated as a snapshot rather than permanent facts. At the time of this review, RoCo-45 is listed at 6.766 and the benchmark RAFT entry at 6.790. Strong named public methods include MEMFOF at 3.289, WAFT-DAv2-a2 at 3.298, DPFlow at 3.442, and SEA-RAFT at 3.686. The same table lists DPFlow without Spring fine-tuning at 4.277 and MEMFOF without it at 3.600.[^spring-leaderboard] Lower is better. Anonymous competition entries rank higher, but their training methods cannot yet be audited from public descriptions.

The controlled lesson is not simply that newer models win. The DPFlow pair directly demonstrates a large gain from Spring-specific supervised fine-tuning. The strongest models also differ in initialization, resolution, frame count, backbone data, and training compute. A fair team experiment should therefore compare:

- the same checkpoint before and after Spring fine-tuning;
- clean accuracy and corruption robustness separately;
- fixed validation sequences and inference resolution;
- reported compute, training images, and pretrained weights.

The benchmark rules permit public external data and test-time adaptation, but require declaring training/pretraining data, checkpoints, and resources. Test labels, manual test annotations, private data, or repeated leaderboard feedback used as training supervision would violate the spirit or rules.[^challenge-rules]

## 5. RobustSpring changes the training objective, not the basic task

RobustSpring applies 20 corruption types spanning color, blur, noise, compression/quality, and weather. The transformations are designed to be consistent across time, stereo views, and depth, which matters because independently corrupting every frame can create artificial apparent motion. The benchmark evaluates both clean accuracy and the change caused by corruptions.[^robustspring]

Its corrupted test images are intentionally not a training set. A robust model should be developed from clean Spring training data, other public training data, and self-created augmentations—not by selecting hyperparameters against RobustSpring test outcomes.[^robustspring]

Three training approaches have credible precedent:

### A. Label-preserving corruption augmentation

Take a labeled clean Spring pair, render a temporally coherent degraded version, and retain the same flow label. Blur, color changes, sensor noise, JPEG artifacts, fog, rain, and snow can be sampled at varied severities. Weather should ideally move consistently in 3D or over time rather than appear independently in the two images. *Distracting Downpour* found that training with non-adversarial synthetic weather improved optical-flow robustness and generalization at little additional cost.[^downpour]

Train on a mixture of clean and corrupted examples. Keeping a substantial clean fraction is important because the challenge score balances clean performance and robustness; optimizing only degraded data can trade one for the other.

### B. Clean-teacher/corrupted-student consistency

Run a frozen or stop-gradient teacher on the clean pair and the student on its degraded counterpart. Train the student with ordinary ground-truth flow loss plus prediction or feature consistency:

\[
L=L_{GT}(S(D(I)),f^*)+\lambda_p L_{pred}(S(D(I)),T(I))+\lambda_f L_{feature}.
\]

Only high-confidence teacher pixels should be used if the teacher prediction replaces a ground-truth target. DistractFlow uses a related idea: it mixes a realistic distractor into one frame, keeps supervised labels for labeled data, and uses confidence-filtered pseudo-label consistency for unlabeled data.[^distractflow] OCAI similarly combines supervised loss with confidence-masked teacher/student pseudo-supervision.[^ocai]

### C. Partial fine-tuning with feature distillation

FlowAnyTime uses a frozen CroCo v2 teacher, a student initialized from the same large pretrained optical-flow model, paired clean/degraded inputs, partial layer unfreezing, and intra-frame/inter-frame feature distillation. Its purpose is to preserve geometric/motion knowledge while learning degradation-insensitive representations for low light, rain, fog, and other conditions.[^flowanytime]

This is closer to modern parameter-efficient robustness adaptation than QLoRA. However, it depends on a much larger pretrained CroCo v2 pipeline and careful paired degradation generation. For this team, the core idea—freeze a reliable teacher and distill clean predictions/features into a corrupted student—is easier to transfer to RAFT/SEA-RAFT/WAFT than to reproduce the paper wholesale.

Unsupervised approaches such as SMURF add full-image warping, photometric and smoothness objectives, self-teaching, and multiple frames so that RAFT can learn from unlabeled video.[^smurf] They become attractive if a large in-domain unlabeled video collection is available. With dense Spring labels already available, they are an optional second phase, not the most direct starting point.

## 6. A realistic plan for two 15-GB T4 GPUs

Two GPUs do not combine into one 30-GB memory pool under normal distributed training. Each GPU holds a model replica, so every sample must fit within 15 GB. Gradient accumulation increases the effective batch but does not reduce the activation memory of one crop. Mixed precision, smaller crops, fewer recurrent iterations, activation checkpointing, and selective freezing are the relevant memory controls.

### Experiment 0: lock the validation and metrics

Hold out complete Spring sequence(s), not random neighboring frames, to avoid near-duplicate temporal leakage. Keep sequence `0022` as the fixed validation sequence if that is the project's current convention. Report at least 1-pixel outlier rate and EPE, with large-motion and unmatched-region slices where the devkit supports them. Never use RobustSpring test submissions as the inner tuning loop.

### Experiment 1: RAFT control

Start with an official Sintel/Things checkpoint. Fine-tune on Spring left-camera forward flow first, using 540×960 random crops, one sample per GPU, AMP, AdamW, gradient accumulation of four, gradient clipping at 1, and OneCycle scheduling. Train all parameters. Try learning rates 1e-5, 3e-5, and 1e-4 rather than assuming the original from-scratch value transfers cleanly.

Use the recurrent sequence loss with 12 training iterations and \(\gamma\) around 0.8–0.85. Compare the current `max_flow=400` validity mask with 1000 or no magnitude cutoff after checking Spring's motion distribution; a 400-pixel mask can silently discard exactly the large motions Spring emphasizes. Validate at a fixed iteration count such as 24 or 32.

The local loader name also matters: if only left/forward data is installed, select the left-specific training split instead of a loader that expects both cameras and directions.

### Experiment 2: modern checkpoint controls

Evaluate released SEA-RAFT, DPFlow, WAFT, and MEMFOF checkpoints without training, at the same input scale and validation split. This determines whether implementation work should go into fine-tuning or robustness augmentation.

For actual T4 fine-tuning, prioritize:

1. RAFT, because the implementation is already integrated and provides a clean experimental control.
2. WAFT-Twins-a2 or another released WAFT variant, because freezing the large backbone and training the head/side path is the most plausible modern memory-conscious approach.
3. A shortened DPFlow Spring continuation, because its official recipe and the value of Spring fine-tuning are documented, while acknowledging that the full 120k-step run is outside the likely budget.
4. SEA-RAFT if it fits at batch one with the selected crop and software stack.
5. MEMFOF primarily for inference or reduced-crop experiments; do not claim reproduction of native-resolution training on T4.

### Experiment 3: corruption-aware supervised fine-tuning

Use only Spring training images to construct a deterministic corruption validation set and stochastic corruptions for training. A strong initial sampling policy is 50% clean, 50% corrupted. Select one corruption family and severity per corrupted example at first; complex mixtures can be added later. Use identical physical/severity parameters across both frames while maintaining realistic temporal evolution for particles.

Continue to train with the real Spring flow label. Measure clean validation after every robustness run. The simplest useful ablation is:

| Run | Supervision | Purpose |
|---|---|---|
| Clean | Ground-truth flow only | Accuracy baseline |
| Augmented | Ground-truth flow on clean/corrupted mix | Test augmentation benefit |
| Consistency | Ground truth + clean-teacher/corrupted-student loss | Test invariance benefit |
| Frozen/partial | Same as consistency, but only selected blocks train | Test memory/overfitting trade-off |

Do not implement all 20 corruption types before the experiment pipeline works. Begin with Gaussian/shot noise, defocus or motion blur, JPEG compression, fog, rain, and snow because they cover qualitatively different failure modes; RobustSpring reports weather among the most damaging categories.[^robustspring]

### Experiment 4: final training and disclosure

After selecting architecture and training recipe using only the fixed validation data, retrain on all permitted Spring training sequences. Record the base checkpoint, every external dataset, augmentation implementation, resolution, crop, iteration count, optimizer, learning-rate schedule, seed, hardware, and test-time processing. This is both scientifically necessary and explicitly relevant to the challenge disclosure rules.[^challenge-rules]

## 7. What not to spend the first week doing

- Do not train RAFT or a modern successor from random initialization on Spring alone. The normal recipe relies on synthetic pretraining and benchmark mixtures.
- Do not attempt the published MEMFOF native-1080p training recipe on a 15-GB T4 and assume gradient accumulation solves per-sample memory.
- Do not call AMP or 8-bit optimizer use “QLoRA.”
- Do not independently randomize spatial transformations of the two frames without transforming the flow vectors correctly.
- Do not tune against RobustSpring's corrupted test images or leaderboard feedback.
- Do not compare models at different resolutions or validation sequences and attribute the difference only to architecture.

## 8. If the scope later expands to stereo or scene flow

The training philosophy stays similar but the labels and architecture change. RAFT-Stereo is normally pretrained on synthetic Scene Flow stereo data and then fine-tuned on target stereo benchmarks; the official repository gives a Middlebury example using a small learning rate, batch two, 384×1000 crops, 22 training iterations, 32 validation iterations, and mixed precision.[^raft-stereo] RAFT-3D learns rigid-motion embeddings/scene flow from FlyingThings3D in its official recipe.[^raft3d]

These are separate models and losses. Ordinary two-frame optical-flow fine-tuning does not automatically produce stereo disparity or 3D scene flow. Establishing one strong optical-flow pipeline first is therefore the lower-risk choice.

## 9. The most defensible answer to “which training should we use?”

Use **supervised checkpoint fine-tuning with dense Spring labels** as the main method. Begin with full RAFT fine-tuning because it is integrated and interpretable. Benchmark modern released checkpoints next. If moving beyond RAFT, try WAFT-style frozen-backbone/head training or a shortened DPFlow/SEA-RAFT continuation. Add **temporally coherent corruption augmentation and clean-teacher/corrupted-student consistency** as the robustness contribution. Treat LoRA/QLoRA as a later ablation only if memory measurements show that trainable parameters—not activations/correlation volumes—are the actual bottleneck.

The exact research query that best captures the problem is:

> Conduct a rigorous review of how state-of-the-art optical-flow models are pretrained and fine-tuned for the Spring and RobustSpring benchmarks. Compare RAFT, SEA-RAFT, DPFlow, MEMFOF, WAFT, and FlowFormer/FlowFormer++, including datasets, initialization, frozen versus trainable modules, losses, crop resolution, batch size, optimization schedule, training compute, and public Spring accuracy. Also review corruption-robust training, clean/degraded teacher–student consistency, self-supervised learning, domain adaptation, and whether LoRA or QLoRA is actually used for optical-flow fine-tuning. Use primary papers, official repositories/configurations, and the official benchmark; distinguish published recipes from practical recommendations for two 15-GB T4 GPUs.

## Sources

[^spring-paper]: [Spring: A High-Resolution High-Detail Dataset and Benchmark for Scene Flow, Optical Flow and Stereo (CVPR 2023)](https://arxiv.org/abs/2303.01943)
[^spring-leaderboard]: [Official Spring optical-flow benchmark](https://spring-benchmark.org/opticalflow?display=accuracy)
[^challenge-rules]: [RoCo Spring 2026 rules and FAQ](https://roco-spring.github.io/rules-faq.html)
[^robustspring]: [RobustSpring: Benchmarking Robustness to Image Corruptions for Optical Flow, Stereo and Scene Flow Estimation](https://arxiv.org/html/2505.09368)
[^raft-paper]: [RAFT: Recurrent All-Pairs Field Transforms for Optical Flow](https://arxiv.org/html/2003.12039)
[^raft-train]: [Official RAFT standard training schedule](https://github.com/princeton-vl/RAFT/blob/master/train_standard.sh)
[^raft-data]: [Official RAFT datasets and augmentation configuration](https://github.com/princeton-vl/RAFT/blob/master/core/datasets.py)
[^sea-paper]: [SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow](https://arxiv.org/html/2405.14793v1)
[^sea-train]: [Official SEA-RAFT staged training script](https://github.com/princeton-vl/SEA-RAFT/blob/main/scripts/train.sh)
[^sea-spring-config]: [Official SEA-RAFT Spring 540×960 training configuration](https://raw.githubusercontent.com/princeton-vl/SEA-RAFT/main/config/train/Tartan-C-T-TSKH-spring540x960-M.json)
[^dpflow-paper]: [DPFlow: Adaptive Optical Flow Estimation with a Dual-Pyramid Framework](https://arxiv.org/html/2503.14880v2)
[^dpflow-config]: [Official DPFlow Spring training configuration in PTLFlow](https://raw.githubusercontent.com/hmorimitsu/ptlflow/main/ptlflow/models/dpflow/configs/dpflow-train4b-spring.yaml)
[^memfof-paper]: [MEMFOF: High-Resolution Training for Memory-Efficient Multi-Frame Optical Flow Estimation (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/papers/Bargatin_MEMFOF_High-Resolution_Training_for_Memory-Efficient_Multi-Frame_Optical_Flow_Estimation_ICCV_2025_paper.pdf)
[^memfof-code]: [Official MEMFOF repository, development branch](https://github.com/msu-video-group/memfof/blob/dev/README.md)
[^waft-paper]: [WAFT: Warping-Alone Field Transforms for Optical Flow](https://arxiv.org/html/2506.21526)
[^flowformer]: [FlowFormer: A Transformer Architecture for Optical Flow](https://arxiv.org/abs/2203.16194)
[^flowformerpp]: [FlowFormer++: Masked Cost Volume Autoencoding for Pretraining Optical Flow Estimation](https://arxiv.org/abs/2303.01237)
[^distractflow]: [DistractFlow: Improving Optical Flow Estimation via Realistic Distractions and Pseudo-Labeling (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Jeong_DistractFlow_Improving_Optical_Flow_Estimation_via_Realistic_Distractions_and_Pseudo-Labeling_CVPR_2023_paper.pdf)
[^ocai]: [OCAI: Improving Optical Flow Estimation by Occlusion and Consistency Aware Interpolation (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Jeong_OCAI_Improving_Optical_Flow_Estimation_by_Occlusion_and_Consistency_Aware_CVPR_2024_paper.pdf)
[^downpour]: [Distracting Downpour: Adversarial Weather Attacks for Motion Estimation (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/html/Schmalfuss_Distracting_Downpour_Adversarial_Weather_Attacks_for_Motion_Estimation_ICCV_2023_paper.html)
[^flowanytime]: [FlowAnyTime: Efficient Fine-tuning with Intra-Inter Frame Distillation for All-Weather Optical Flow Estimation (AAAI 2026)](https://ojs.aaai.org/index.php/AAAI/article/view/38019)
[^smurf]: [SMURF: Self-Teaching Multi-Frame Unsupervised RAFT with Full-Image Warping (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/papers/Stone_SMURF_Self-Teaching_Multi-Frame_Unsupervised_RAFT_With_Full-Image_Warping_CVPR_2021_paper.pdf)
[^lora]: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
[^qlora]: [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
[^raft-stereo]: [Official RAFT-Stereo repository](https://github.com/princeton-vl/RAFT-Stereo)
[^raft3d]: [Official RAFT-3D repository](https://github.com/princeton-vl/RAFT-3D)
