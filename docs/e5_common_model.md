# E5-G / E5-S common model

## Controlled variable

E5-G and E5-S share the same images, labels, source-id split, YOLO11n-OBB
checkpoint, frozen OpenCLIP text tower, P5 FiLM adapter, detection loss, optimizer
settings, and evaluation sets. The prompt policy is the only intended difference:

- E5-G maps every image to one generic lighting prompt.
- E5-S maps `transform_name` to one of four photometric prompts.

The prompts describe brightness, contrast, and color temperature only. They do
not claim part count, rotation, or position because those factors were not varied
or recorded by the post-processing pipeline.

## Visual detector

The detector is `yolo11n-obb.pt` at 1280 pixels with one class, matching E0. Ultralytics
YOLO11-OBB produces P3, P4, and P5 features and uses the standard eight-coordinate
OBB label representation. The model is initialized from the same checkpoint used
by both experiments.

## Frozen text encoder

OpenCLIP `ViT-B-32` with `laion2b_s34b_b79k` weights maps each prompt to a
512-dimensional vector. All OpenCLIP parameters are frozen and the encoder stays
in evaluation mode. Because E5 uses at most five unique prompts, encode each
unique string once and keep a prompt-to-vector cache; do not run OpenCLIP for
every training batch. The encoder can be released after the cache is built to
avoid keeping the unused OpenCLIP visual tower in GPU memory.

If the server cannot download the OpenCLIP checkpoint, download it on a connected
machine, copy it to the server, and set `model.text_encoder.pretrained` to that
absolute local checkpoint path. Do not replace private-model downloads with an
untrusted third-party proxy.

## P5 FiLM adapter

FiLM is installed as a forward hook on module 10, the backbone P5 C2PSA output in
the official YOLO11-OBB architecture. The feature channel count is inferred by a
64-pixel dummy forward pass, so the adapter does not hard-code the width-scaled
YOLO11n channel count. The 64-pixel probe only discovers channels; actual training
and validation remain at 1280 pixels.

For image feature `F` and normalized text vector `t`, a trainable linear layer
produces `gamma(t)` and `beta(t)` and applies:

```text
F_conditioned = F * (1 + gamma(t)) + beta(t)
```

The projection weight and bias start at zero. The first forward pass therefore
leaves YOLO features unchanged, while gradients can immediately update the FiLM
projection. Only YOLO and FiLM parameters enter the optimizer.

Install FiLM before the Ultralytics trainer constructs its optimizer. Otherwise
the newly attached projection will not be present in the optimizer parameter
groups. When loading an E5 checkpoint, reinstall the adapter and its forward hook
before loading the E5 state dictionary.

## Augmentation policy

Disable Mosaic, MixUp, CopyPaste, HSV, and BGR augmentation in both experiments.
Mosaic would combine images carrying different photometric prompts, while HSV
would change the described brightness or color after the prompt was selected.
Geometry-only augmentation may remain enabled because the prompts make no claim
about position or rotation and Ultralytics updates the OBB labels accordingly.

This augmentation policy differs from a normal YOLO baseline, so comparisons
against E0 require an E5-A image-only control trained with the same disabled
augmentations. E5-G versus E5-S itself remains a clean prompt-policy comparison.

## Trainer integration contract

The training entry point still needs to perform these operations for every batch:

1. Resolve each `im_file` to its manifest row.
2. Select the generic prompt for E5-G or the row's photometric prompt for E5-S.
3. Stack cached embeddings in exactly the image order of the batch.
4. Call `film.set_text_features(batch_embeddings)`.
5. Run the normal Ultralytics OBB forward/loss/backward step.
6. Clear or replace the text features before the next batch.

Validation must use the same mechanism. Real images have no measured per-image
photometric parameters, so the primary E5-S protocol uses the fixed neutral
evaluation prompt declared in the configuration. Do not select a prompt after
looking at real-test metrics.

## Training

The E5 entry point subclasses Ultralytics `OBBTrainer`. Before the optimizer is
built, it installs FiLM and builds the frozen OpenCLIP prompt cache. For each
training batch, `im_file` is resolved to a manifest row, its cached embedding is
placed on the detector device, and the normal YOLO11-OBB detection loss performs
backpropagation through YOLO and FiLM only. Validation uses the fixed evaluation
prompt stored in the checkpoint.

Run a small smoke test first:

```bash
bash scripts/train_e5.sh configs/experiment/e5_g.yaml \
  --epochs 1 --batch-size 2 --workers 0 --fraction 0.03
bash scripts/train_e5.sh configs/experiment/e5_s.yaml \
  --epochs 1 --batch-size 2 --workers 0 --fraction 0.03
```

Then run seed 42:

```bash
bash scripts/train_e5.sh configs/experiment/e5_g.yaml
bash scripts/train_e5.sh configs/experiment/e5_s.yaml
```

## Model check

After installing the environment and making the pretrained weights available:

```bash
python tools/check_e5_model.py --config configs/experiment/e5_g.yaml
python -m pytest -q
```

The check verifies the detector, OpenCLIP embedding dimension, P5 feature channel
count, identity-initialized FiLM path, and a two-image conditioned forward pass.

## Counterfactual prompt validation

Use one trained E5-S checkpoint and the same synthetic validation split for all
three conditions. `correct` uses each row's recorded transform, `fixed` uses the
neutral prompt for every image, and `shuffled` applies a deterministic balanced
permutation in which every transform receives a wrong prompt.

```bash
bash scripts/validate_e5_counterfactual.sh \
  /data/huangwenwen/yolo11/outputs/E5/E5-S_seed42/weights/best.pt \
  --batch-size 2 --workers 4 --device 0
```

The summary is written to:

```text
/data/huangwenwen/yolo11/outputs/E5/counterfactual/
  E5-S_seed42/counterfactual_metrics.csv
```

Text semantics are supported when the same checkpoint performs better with
`correct` prompts than with both `fixed` and `shuffled` prompts. Repeat the
comparison across the planned training seeds before treating the effect as stable.
