# System Architecture

## End-to-End Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│                          USER INTERFACE                              │
│  Gradio web app (HuggingFace Spaces) — webcam capture OR image upload│
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       PREPROCESSING                                  │
│  MediaPipe face detector → bounding box → crop with margin →         │
│  resize to 224×224 → normalize (DINOv2 stats)                        │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  DINOv2-small BACKBONE  (21M params)                 │
│  • Layers 0-8 FROZEN (generic visual features)                       │
│  • Layers 9-11 FINE-TUNED (task specialization)                      │
│  • Output: 384-d CLS token + patch features                          │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      SHARED MLP TRUNK                                │
│  Linear(384 → 256) → LayerNorm → GELU → Dropout(0.3)                 │
│  Linear(256 → 256) → LayerNorm → GELU → Dropout(0.3)                 │
└──────────────────────┬──────────────────────┬────────────────────────┘
                       │                      │
                       ▼                      ▼
            ┌─────────────────┐    ┌────────────────────┐
            │   STRESS HEAD   │    │   FATIGUE HEAD     │
            │   BN → Drop →   │    │   BN → Drop →      │
            │   Linear(256→3) │    │   Linear(256→2)    │
            └────────┬────────┘    └─────────┬──────────┘
                     │                       │
                     ▼                       ▼
              softmax → probs         softmax → probs
                     │                       │
                     └───────────┬───────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│             LLM EXPLANATION LAYER  (Claude API)                      │
│  Engineered prompt = system role + 3 few-shot examples +             │
│                       constraints + current prediction               │
│  Temperature 0.7, max 200 tokens                                     │
│  Fallback: canned response if API unavailable                        │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 ▼
                       Wellness recommendation
                       (2-3 sentence natural language)
```

## Data flow during training

```
Kaggle dataset mounts (/kaggle/input/...)
       │
       ▼
Symlink into /kaggle/working/data/processed/{drowsiness,fer2013}
       │
       ▼
PyTorch Dataset classes (DrowsinessDataset, FER2013Dataset)
       │
       ├─► subject-grouped 70/15/15 split (no leakage)
       │
       ▼
CombinedFacialDataset (unified -1-as-missing label format)
       │
       ▼
DataLoader (batch=64 on TPU, 32 on GPU)
       │
       ▼
Train loop in Trainer:
   ├─► AMP forward (bf16 on TPU / fp16 on GPU)
   ├─► MultiTaskLoss (weighted CE + focal)
   ├─► AdamW with discriminative LR + cosine warmup
   ├─► grad clip 1.0
   ├─► W&B logging + TensorBoard
   └─► best checkpoint on val/balanced_acc_mean
       │
       ▼
   Save to HuggingFace Hub
       │
       ▼
   HF Space pulls model on next deployment
```

## CI/CD Architecture

```
Developer push to main
        │
        ▼
  ┌─────────────────────────────────────────┐
  │  GitHub Actions: ci.yml                 │
  │  • pytest                               │
  │  • ruff lint                            │
  │  • structure checks                     │
  └────────────┬────────────────────────────┘
               │ pass
               ▼
  ┌─────────────────────────────────────────┐
  │  GitHub Actions: deploy_hf.yml          │
  │  • git push to HF Spaces repo           │
  │  • Space auto-rebuilds container        │
  └────────────┬────────────────────────────┘
               │
               ▼
        Live app updated

  Separately, on schedule:
  ┌─────────────────────────────────────────┐
  │  GitHub Actions: retrain.yml (cron)     │
  │  • check drift via Evidently            │
  │  • open issue if drift > threshold      │
  │  • human runs Kaggle TPU notebook       │
  │  • new model pushed to HF Hub           │
  └─────────────────────────────────────────┘
```

## Parameter budget

| Component | Total params | Trainable | Frozen |
|-----------|-------------|-----------|--------|
| DINOv2-small backbone | 21.0 M | ~5.2 M (last 3 layers) | ~15.8 M |
| Shared MLP trunk | 0.17 M | 0.17 M | 0 |
| Stress head | 0.001 M | 0.001 M | 0 |
| Fatigue head | 0.0005 M | 0.0005 M | 0 |
| **Total** | **~21.2 M** | **~5.4 M** | **~15.8 M** |

Roughly 25% of parameters are trainable — small enough to fit in TPU memory at batch 64.
