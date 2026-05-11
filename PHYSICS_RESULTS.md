# Physics Verification — What We Built, What We Found, How It Helps Classification

## 1. The problem in one paragraph

You have N IMU sensors stuck on a body. A neural classifier looks at each sensor's data (rotation + acceleration over a 300-frame window) and predicts which of 24 SMPL body regions the sensor is on (pelvis, l_hip, r_hip, l_thigh, …, r_hand). It's right ~87% of the time. **Most of the remaining 13% errors are systematic** — the classifier consistently confuses certain pairs. The most common confusions are along the arms (forearm↔upper_arm, hand↔forearm), and across the body's left/right symmetry (l_hip↔r_hip, l_thigh↔r_thigh).

The goal of the physics work is to add a **second opinion** that uses biomechanics — a "physics scorer" that looks at the same IMU data and says "this combination of region labels is/isn't physically plausible." When the classifier is unsure between two predictions, the physics scorer breaks the tie.

## 2. Why physics can help where the classifier struggles

The classifier looks at one sensor at a time. It can tell a forearm from a head easily (very different motion profiles), but it can't tell a left forearm from a right forearm — they look identical in isolation, the body is symmetric.

Physics knows something the classifier doesn't: **how joints connect**. If sensor #1 is on l_upper_arm and sensor #2 is on l_forearm, the rotation between them must look like a real human elbow — bending the right way, within human range of motion. If you swap the labels (claim #1 is r_upper_arm, #2 is l_forearm), now you're claiming a "joint" between two regions that aren't actually connected — the rotation between them won't look like any real joint, and the physics scorer should reject it.

So physics turns a hard single-sensor problem into a much easier multi-sensor consistency check.

## 3. What was planned (`PHYSICS_VERIFICATION_PLAN.md` in short)

Three independent physics scorers, combined into one:

1. **Kinematic scorer** — for every pair of sensors that are claimed to be on connected body parts (parent–child in the SMPL skeleton), compute the rotation between them and check it against a precomputed "envelope" of plausible joint angles learned from the training data. Score = how well the rotations stay inside the envelope.
2. **Gravity scorer** — during still moments, the IMU's acceleration channel should point at gravity. Each body region has a typical orientation w.r.t. gravity (e.g., the foot's "up" is different from the head's "up"). If the sensor's gravity direction doesn't match the claimed region, the score drops.
3. **Acceleration scorer** — different body regions accelerate differently (your hand swings around a lot, your pelvis doesn't). Compare the sensor's acceleration statistics against the typical statistics for the claimed region.

The idea was to combine all three with tunable weights and run the result on classifier candidates to rerank them.

### What changed during implementation

- **Gravity scorer is dead on this dataset.** The IMU acceleration channel in our preprocessed data is *linear acceleration only* — gravity is already subtracted. There's nothing for the gravity scorer to recover. We left the code in place (in case we ever switch to raw IMU data), but its weight is set to 0.
- **Kinematic scorer was buggy at first.** It loaded the joint-angle envelope file but didn't actually use it — it just scored "how far is the relative rotation from identity," which has no biological meaning. Fixed: it now actually checks each parent–child rotation against the per-pair envelope.
- **Joint-angle envelope was undertrained.** Originally computed from only the first frame of each motion segment (basically T-pose). Fixed: now uses all 300 frames per segment.
- **Rotation math was fragile.** The 6D-to-rotation-matrix function didn't orthogonalize properly. Worked fine on clean SMPL inputs, but fragile to anything else. Fixed with proper Gram-Schmidt.

So today, of the three scorers, **kinematic and acceleration are the two real ones**; gravity is inert.

## 4. What the kinematic scorer actually does (plain English)

Given a candidate "combo" — a list of sensors with claimed region labels:

1. Walk through every sensor in the combo. Look up its claimed region's parent in the SMPL skeleton (e.g., l_forearm's parent is l_upper_arm). If the parent's region is also claimed by some other sensor in the combo, you have a parent–child pair to check.
2. For each such pair, compute the rotation between the parent sensor and the child sensor over all 300 frames.
3. Look up that pair's joint-angle envelope (precomputed from training data — the 5th and 95th percentiles of the rotation observed for that parent–child pair).
4. Score: each frame inside the envelope contributes 1.0; frames outside decay smoothly toward 0 the further outside they go.
5. Average over frames and over all parent–child pairs in the combo.

If a combo has no parent–child pairs (e.g., three sensors on completely disconnected body parts), the scorer has nothing to say — it returns a default 0.5.

**Crucial implication:** the kinematic scorer can only fire when the combo includes at least two sensors on the same kinematic chain. If your sensor placement is on five disconnected body parts, kinematic physics gives you no signal.

### The kinematic chains (SMPL skeleton, 24 regions)

This is the parent–child tree the scorer uses. Two sensors must lie on a path through this tree (parent–child, grandparent–grandchild, etc.) for the scorer to have anything to say about them. Source: `smpl_regions.py:REGION_PARENTS`.

```
pelvis (root)
├── spine_lower
│   └── spine_mid
│       └── spine_upper
│           ├── neck
│           │   └── head
│           ├── l_collar
│           │   └── l_shoulder
│           │       └── l_upper_arm
│           │           └── l_forearm
│           │               └── l_hand
│           └── r_collar
│               └── r_shoulder
│                   └── r_upper_arm
│                       └── r_forearm
│                           └── r_hand
├── l_hip
│   └── l_thigh
│       └── l_shin
│           └── l_foot
└── r_hip
    └── r_thigh
        └── r_shin
            └── r_foot
```

There are **five distinct chains** descending from the pelvis:

| Chain | Regions (root → leaf) |
|---|---|
| Spine + neck | pelvis → spine_lower → spine_mid → spine_upper → neck → head |
| Left arm | pelvis → spine_lower → spine_mid → spine_upper → l_collar → l_shoulder → l_upper_arm → l_forearm → l_hand |
| Right arm | pelvis → spine_lower → spine_mid → spine_upper → r_collar → r_shoulder → r_upper_arm → r_forearm → r_hand |
| Left leg | pelvis → l_hip → l_thigh → l_shin → l_foot |
| Right leg | pelvis → r_hip → r_thigh → r_shin → r_foot |

Two sensors on the **same** chain → kinematic scorer can compare them. Two sensors on **different** chains share at most a high-up common ancestor (e.g., l_forearm and r_forearm only share spine_upper and above), so unless that ancestor is also in the combo, the scorer is silent across chains.

**Why the L/R confusions are hard for physics:** the left and right legs are mirror chains rooted at the same pelvis. The pelvis→l_hip and pelvis→r_hip joints have nearly identical envelopes by anatomical symmetry — physics alone cannot tell a left hip from a right hip. The discrimination only kicks in one level deeper, at the hip→thigh joint, because "right hip connected to left thigh" is anatomically impossible. The same logic applies to arms (collar→shoulder is symmetric, shoulder→upper_arm is decisive).

**Why arm confusions are easy for physics:** every arm confusion in the top errors (forearm↔upper_arm, forearm↔hand, hand↔upper_arm) is between *adjacent* regions on the same chain. Swapping their labels rewires the chain, and the resulting joint angles fall outside the trained envelope.

## 5. What we just measured: the per-pair discrimination test

The classifier's confusion matrix (`results/penalty_sweep/eval_best_final/confusion_aggregate.json`) gives us a ranked list of which region pairs the classifier confuses most. Top offenders:

- Arm pairs: `r_forearm↔r_upper_arm` (350 errors), `l_forearm↔l_upper_arm` (259), `l_forearm↔l_hand` (223), `r_forearm↔r_hand` (204).
- Left/right pairs: `l_hip↔r_hip` (134), `l_thigh↔r_thigh` (80).

The question we asked: **for each of these confusable pairs, can the kinematic scorer tell which label is correct, given that the data really came from one of them?**

Setup: take real data for region A. Build two candidate combos that contain it — one labeled correctly (with A), one labeled wrong (with B in A's slot). Score both. If the scorer is doing its job, the correct labeling should score higher.

We measured this in two regimes:

- **Best case** — the combo also contains A's parent and one child of A. This is the regime where the kinematic scorer has every signal it could possibly use.
- **Realistic** — the combo contains A plus 1–4 random other sensors. This estimates how often the scorer fires when sensor placement is arbitrary.

### Headline numbers

| Regime | Errors recoverable (non-L/R, of 1522) | Errors recoverable (L/R, of 243) |
|---|---|---|
| Best case (kinematic neighbors present) | ~1476 (97%) | ~243 (~100%) |
| Realistic n=3 (random co-sensors) | ~249 (16%) | ~61 (25%) |

Train and test splits agree within 1 percentage point, so this isn't memorization.

### Two kinds of wins, important to understand

The best-case "97% recoverable" number breaks into:

1. **Structural wins** — most cases. Swapping A for B turns a valid kinematic chain into garbage with no parent–child relationships at all. The scorer literally has nothing to score for the swap, and falls back to 0.5. Truth scores ~0.85, swap scores 0.5 → easy win, but it's a topological win, not an envelope-quality win.
2. **Envelope-quality wins** — the strong cases. Both combos produce valid parent–child pairs, but the truth's rotations fit the envelope and the swap's don't. Examples: `l_hip↔r_hip` with pelvis+thigh in combo (truth scores 0.86, swap 0.10 — enormous gap). All the L/R pairs that include a child-side sensor behave this way.

Both are useful for a reranker, but the second is the kind that generalizes most reliably.

### Direct illustration of the user's example

> "Combo: l_shoulder, r_hip, neck — but the data is really l_shoulder, l_hip, neck. Can the kinematic scorer fix it?"

**Not in that combo.** Neither l_shoulder nor neck is on l_hip's kinematic chain (l_hip's parent is pelvis, child is l_thigh, neither in the combo). Both labelings produce zero parent–child pairs, both score 0.5, scorer is silent.

**Yes, if the combo also contained pelvis and a thigh sensor.** Then the scorer compares (pelvis→l_hip) vs (pelvis→r_hip) — these envelopes are similar by symmetry, so that pair alone is ambiguous (~0.55 AUC). But the (l_hip→l_thigh) vs (r_hip→l_thigh) check is decisive: a "right-hip-to-left-thigh" connection isn't a real anatomical joint and produces nonsense rotations. Truth wins with ~0.77 score gap, AUC 1.0.

This is the central lesson: **the physics scorer's power depends entirely on whether the surrounding sensors give it kinematic context.**

## 6. What this means for your classification pipeline

The end goal is to fix as many of those 2,931 errors as possible. Here's how the physics evidence shapes the approach:

### What works (highest-leverage pairs)

- **Arm chains** account for ~1,200 of the 2,688 non-L/R errors. Every confusion in this group (forearm↔upper_arm, forearm↔hand, hand↔upper_arm, upper_arm↔shoulder) is a parent–child or grandparent confusion on the same chain. **If the test setup includes 2+ sensors on the same arm**, the kinematic scorer can almost perfectly disambiguate them. Even with random co-sensors, ~20% of windows include a same-chain neighbor, recovering ~10–20% per pair.
- **L/R hip and thigh confusions** (214 of 243 L/R errors) are recoverable as long as a child-side sensor is in the combo (l_thigh or r_thigh, l_shin or r_shin). Pure pelvis+hip cannot tell left from right; pelvis+hip+thigh can, decisively.

### What doesn't work (low-yield pairs)

- **Off-chain confusions** like `l_foot→r_shin`, `r_hip→r_shoulder`, `r_thigh→spine_upper`. These are kinematically nonsensical, so any combo that triggers an envelope check will reject them — but that's a structural rejection, not envelope quality. Counts are small (~16–33 errors each) so absolute impact is limited.
- **Spine and head confusions** (`head↔neck`, `head↔spine_upper`). Few errors, modest physics signal.

### What the physics scorer cannot fix on its own

Anything where the combo contains only one sensor on a given kinematic chain. The scorer is silent in those cases (75–90% of random combos at n_ctx=3). To turn this from "occasionally helpful" into "consistently helpful," the integration needs to **deliberately include kinematic neighbors when scoring candidates** — not rely on the test setup happening to include them.

### Recommended integration path (concrete)

1. Run the classifier on each sensor in the multi-sensor test setup to get top-k region predictions per sensor (e.g., top 3).
2. Enumerate candidate combos by taking the cross-product of top-k labels across sensors (3^5 = 243 combos for 5 sensors — trivial).
3. For each combo, run the kinematic scorer.
4. If the combo has no parent–child pairs at all, the kinematic scorer is silent — fall back to the classifier's output.
5. If the kinematic score discriminates between two top combos, prefer the higher-scoring one.

The acceleration scorer (the second non-dead component) can be added as a second voice for cases where kinematic is silent. Per the original plan and the existing physics_sweep results, accel discriminates roughly as well as kinematic against random combos (AUC ~0.87) but much worse against L/R swaps (AUC ~0.55) — so kinematic is the L/R discriminator, accel is the off-chain discriminator. Use them both.

### Realistic ceiling on accuracy improvement

Translating "errors recoverable" into accuracy gain on the 21,696-window test set:

- **Realistic regime, kinematic alone, n_ctx=3 random**: ~310 errors recovered → 87.5% → 88.9% (+1.4 pp).
- **Best case (combos seeded with kinematic neighbors), kinematic alone**: ~1700 errors recovered → 87.5% → 95.4% (+7.9 pp).
- Add the acceleration scorer for off-chain confusions and L/R-without-thigh cases: another modest gain on top.

The 7.9 pp gain is the right thing to chase. The 1.4 pp gain is what you get if you wire the scorer up naively without curating the candidate combos. The difference is entirely about *whether the candidate combos include kinematic neighbors* — that's the design lever.

## 7. Files in this work

- `PHYSICS_VERIFICATION_PLAN.md` — full plan, what was tried, what was fixed (sections 9.1–9.7 are the implementation log).
- `verify_combo.py` — the four scorers (kinematic, gravity, accel, combined).
- `compute_training_stats.py` — builds joint-angle envelopes, accel distributions from training data.
- `extract_calibration.py` — pulls sensor-to-body rotation matrices from the dataset generator.
- `physics_sweep.py` — generic diagnostic: AUC vs random / off-chain / L/R swaps across n_sensors ∈ {2,3,4,5}.
- `pair_discrimination.py` — the per-confusion-pair test described above.
- `stats/joint_angle_limits.npy`, `stats/accel_distributions.npy`, `calibration/region_sensor_rotmats.npy` — the precomputed artifacts the scorers consume.
- `stats/sweep_report.json` — generic sweep results.
- `stats/pair_discrimination_train.json`, `stats/pair_discrimination_test.json` — per-pair numbers, train and test.

## 8. One-sentence summary

**The kinematic scorer is a strong second opinion that turns most of the classifier's hardest confusions (arm chains and left/right symmetry) from coin flips into near-certain calls — but only when the candidate combos it scores include kinematic neighbors. Build the reranker to seed those neighbors deliberately, and you can plausibly take overall accuracy from 87.5% toward ~95%.**
