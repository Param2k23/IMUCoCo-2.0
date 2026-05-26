# Learned Joint Relationship Model (LJRM)

## Overview

The LJRM learns a **64-dimensional embedding** that captures the kinematic relationship between any pair of body joints, trained contrastively using the SMPL skeleton's adjacency graph as supervision signal. It replaces hand-crafted kinematic features with a learned representation.

---

## Architecture

The model has two sub-networks stacked together:

### 1. `JointEncoder` — per-joint temporal embeddings

Takes a window of IMU data for a single joint and produces a compact embedding.

```
Input:  [B, T=300, 9]   ← (batch, time, 9-channel virtual IMU)
   ↓  Linear projection: 9 → 128 dims
   ↓  Sinusoidal positional encoding
   ↓  Transformer encoder (2 layers, 4 heads, FFN dim=512)
   ↓  Mean pool over time → [B, 128]
   ↓  Linear projection: 128 → 64
   ↓  L2 normalize
Output: [B, 64]          ← unit-norm embedding
```

### 2. `PairRelationshipHead` — joint-pair relationship vector

Takes two joint embeddings and fuses them into a relationship descriptor.

```
Input:  e_i [B, 64],  e_j [B, 64]
   ↓  Concatenate [e_i | e_j | |e_i − e_j| | e_i ⊙ e_j] → [B, 256]
   ↓  MLP: 256 → 256 (ReLU) → 64
   ↓  L2 normalize
Output: [B, 64]          ← relationship embedding for pair (i, j)
```

The four-way fusion (concat, difference, element-wise product) gives the head explicit access to both individual identity and how the two joints differ/relate.

---

## Training Signal — NT-Xent Contrastive Loss

The model is trained with no explicit labels. Instead, the **SMPL skeleton graph** (24 joints, 23 kinematic edges) defines what counts as a "positive pair":

- Two joint pairs that are **kinematically adjacent** in the skeleton are pulled together in embedding space
- All other pairs in the batch are pushed apart
- Temperature τ = 0.07 makes the loss sharp

This forces the model to encode **structural body knowledge** — shoulder↔elbow should embed similarly to hip↔knee because both are adjacent limb segments.

---

## Training Setup

| Setting | Value |
|---|---|
| Data | DIP-IMU, 885 sequences × 300-frame windows |
| Optimizer | AdamW (lr=3e-4, wd=1e-4) |
| Scheduler | CosineAnnealingLR (30 epochs) |
| Gradient clipping | max norm = 1.0 |
| Best checkpoint | Epoch 22, val_loss = 4.193 |

---

## Inference — Building the Lookup Table

After training, the model runs once over a sequence to build a `[24, 24, 64]` lookup table:

- For every joint pair (i, j), compute `PairRelationshipHead(JointEncoder(joint_i), JointEncoder(joint_j))`
- Store the 64-D vector at position `[i, j]`
- This table is used downstream for efficient pair-relationship lookup without re-running the full model

---

## Performance

On a 276-class pair-retrieval task (given a query pair, find the matching pair from 276 candidates):

| Metric | Score | Chance baseline |
|---|---|---|
| Top-1 accuracy | 21.80% | 0.36% |
| Top-5 accuracy | 59.59% | 1.81% |
| Top-10 accuracy | 76.37% | 3.62% |
| MRR | 0.389 | — |

Adjacent pairs (40% top-1) are retrieved more reliably than non-adjacent pairs (20% top-1), which makes sense — the contrastive loss explicitly uses adjacency as its signal.

---

## Key Files

- `aliasgars_work/training.ipynb` — model definition and training loop
- `aliasgars_work/initial_testing.ipynb` — evaluation and retrieval metrics
- `aliasgars_work/joint_rel_model_best.pt` — best saved checkpoint
