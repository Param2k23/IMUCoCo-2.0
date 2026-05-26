# Joint Relationship Lookup Table

A lookup table encodes the **pairwise relationship between every pair of the 24 SMPL body joints** as a fixed vector. At inference time an unknown sensor configuration is identified by matching its observed joint-pair signals against this pre-built gallery.

---

## Table Structure

The table is a tensor of shape **`[24, 24, F]`**, where axis 0 and 1 index the two joints and axis 2 holds the `F`-dimensional relationship descriptor. Two variants exist:

| Variant | `F` | Stored files |
|---------|-----|--------------|
| Handcrafted | 6 | `lookup_table_mean.pt`, `lookup_table_std.pt`, `lookup_table_mean_normalized.pt` |
| Learned | 64 | `notebooks/joint_rel_model_best.pt` |

Only the 276 unique unordered pairs (upper triangle) carry meaningful content; the diagonal is zero.

---

## Features

### Handcrafted (6-D)

Each feature is derived from a 300-frame window of virtual-IMU data per joint pair `(i, j)`.  
Acceleration channels 6–8 and orientation channels 0–5 of the 9-channel signal are used.

| # | Name | Signal | Description |
|---|------|--------|-------------|
| 0 | `cross_corr_peak` | accel | Peak value of the normalised cross-correlation between joints i and j |
| 1 | `cross_corr_lag` | accel | Lag at that peak, normalised by window length T |
| 2 | `magnitude_ratio` | accel | log(RMS_i / RMS_j), clipped to [−5, 5] |
| 3 | `dtw_distance` | accel | DTW distance on a 30-point downsample, normalised by T |
| 4 | `orientation_similarity` | orient | Mean cosine similarity between orientation vectors over time |
| 5 | `relative_orientation_std` | orient | Standard deviation of that cosine similarity over time |

### Learned (64-D)

A shared **Transformer encoder** (`JointEncoder`) maps each joint's `[T, 9]` time-series to a 64-D unit vector. A **`PairRelationshipHead`** MLP then combines the two per-joint embeddings into a single 64-D relationship vector:

```
combined = [e_i ; e_j ; |e_i − e_j| ; e_i ⊙ e_j]  →  Linear(256 → 64)  →  L2-normalise
```

---

## How the Table is Generated

### Handcrafted

For every training sequence (885 files) and every joint pair, the 6 features above are computed from a 300-frame window. The per-pair vectors are averaged across all sequences. The resulting mean table is then **min-max normalised** per feature dimension to `[0, 1]` using the training-data range; the raw mean and its per-entry standard deviation are also saved for diagnostics.

### Learned

1. The model is trained with **NT-Xent contrastive loss**: kinematically adjacent joint pairs (edges of the SMPL skeleton graph) are positive pairs; all other pairs within the batch are negatives.
2. After training, a gallery is built by running the model over all 885 training sequences and accumulating the per-pair 64-D embeddings with **Welford's online algorithm** (constant memory regardless of dataset size).
3. The final table is the per-pair mean embedding; a companion std tensor captures cross-sequence consistency.

---

## How the Table is Used

At test time, given a window of IMU data from `N` sensors:

1. Compute the relationship descriptor for every observed sensor pair `(i, j)` — either the 6 handcrafted features or the 64-D learned embedding.
2. Look up the closest entry in the pre-built gallery using **negative L² distance** (handcrafted) or **cosine similarity** (learned).
3. The top-ranked gallery entry identifies which joint pair `(i, j)` each sensor pair corresponds to, resolving the sensor-to-body-location assignment.

For `N`-sensor **configuration identification**, all `C(N, 2)` pair descriptors are aggregated; the configuration whose gallery entries best match the query is selected (summing pair-wise similarity scores, with permutation-invariant scoring to handle unknown sensor ordering).

---

## Retrieval Metrics

Evaluated on 19 held-out test sequences. Chance baseline for all-pairs retrieval: **Top-1 = 0.36%** (1 in 276).

### All-pairs retrieval (276 classes, 5 244 queries)

| Metric | Handcrafted | Learned | Improvement |
|--------|-------------|---------|-------------|
| Top-1  | 4.12 % | **21.80 %** | 5.3× |
| Top-5  | 14.23 % | **59.59 %** | 4.2× |
| Top-10 | 23.99 % | **76.37 %** | 3.2× |
| MRR    | 0.109 | **0.389** | 3.6× |

### N-sensor configuration identification (learned, permutation-invariant)

| N sensors | Configs | Top-1 | Chance | Lift | MRR |
|-----------|---------|-------|--------|------|-----|
| 2 | 136 | 14.86 % | 0.74 % | 20× | 0.286 |
| 3 | 680 | 8.00 % | 0.15 % | 54× | 0.174 |
| 4 | 2 380 | 3.86 % | 0.04 % | 92× | 0.092 |
| 5 | 6 188 | 1.75 % | 0.02 % | 108× | 0.051 |

### Hardest / easiest pairs (learned, per-pair Top-1)

- **Easiest**: `right_collar ↔ right_shoulder` (89.5 %), `right_collar ↔ head` (84.2 %), `left_collar ↔ head` (84.2 %)
- **Hardest**: lower-body pairs such as `pelvis ↔ right_ankle`, `right_knee ↔ right_ankle` (0 %) — these joints share similar motion statistics
- **Adjacent pairs** average Top-1: **40 %** vs **non-adjacent**: 20 %
