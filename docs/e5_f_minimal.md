# E5-F minimal shift-alignment experiment

E5-F inherits E5-S and changes only `loss.shift_weight` plus the batch grouping
needed to calculate that loss. Every source contributes its identity image and
three photometric variants once per epoch. Images, labels, prompts, detector,
FiLM location, augmentations, optimizer settings, image size, and seed remain
the same as E5-S.

For each source group, FiLM-conditioned P5 is globally pooled and projected to
512 dimensions. The three visual shifts from `identity` to each transformed
image are aligned with the matching frozen CLIP text shifts:

```text
L = L_det + 0.01 * mean(1 - cosine(delta_visual, delta_text))
```

## Run

Smoke test:

```bash
bash scripts/train_e5.sh configs/experiment/e5_f.yaml \
  --epochs 1 --batch-size 4 --workers 0 --fraction 0.03 \
  --name E5-F-smoke_seed42
```

Formal run:

```bash
bash scripts/train_e5.sh configs/experiment/e5_f.yaml
```

The batch size must be divisible by four. The formal inherited batch size is
16. Training writes the normal `results.csv` plus `shift_audit.csv` containing:

- mean shift loss and visual-text shift cosine per epoch;
- mean number of valid pairs per batch;
- FiLM gradient norm;
- shift-projection gradient norm.

## Counterfactual and continuation gate

Run the same correct/fixed/shuffled validation on the E5-F checkpoint:

```bash
bash scripts/validate_e5_counterfactual.sh \
  /data/huangwenwen/yolo11/outputs/E5/E5-F-lambda0.01_seed42/weights/best.pt \
  --config configs/experiment/e5_f.yaml --batch-size 4 --device 0
```

Then evaluate the predefined gates, using the E5-S correct-prompt mAP50-95 as
the baseline:

```bash
python -m spg_lcda.evaluate_e5_f_gate \
  --audit /data/huangwenwen/yolo11/outputs/E5/E5-F-lambda0.01_seed42/shift_audit.csv \
  --counterfactual /data/huangwenwen/yolo11/outputs/E5/counterfactual/E5-F-lambda0.01_seed42/counterfactual_metrics.csv \
  --baseline-map5095 0.8904102848388483
```

The script writes `continuation_decision.json`. Continue to formal real-domain
validation only when all automatic gates pass:

- FiLM and shift-head gradient norms are non-zero;
- final three-epoch shift cosine exceeds the first three epochs by at least 0.10;
- correct-prompt mAP50-95 exceeds both fixed and shuffled prompts by at least 0.005;
- synthetic correct-prompt mAP50-95 drops by no more than 0.005 from E5-S.

After these gates pass, the final manual continuation condition is an average
real-domain mAP50-95 gain of at least 0.01 over E5-S on the unchanged A69, B79,
and tiaocai tests. Otherwise stop this data route and build `S_pair-v2`.
