#  Limitations: Calibration Under Distribution Shift & Drift

> **Part of:** *Can You Trust Post-Hoc Calibration Under Shift? — Revisiting the Ovadia Benchmark with Modern Calibration Methods*

---

##  Overview

This section documents the **known limitations** of all four post-hoc calibration methods evaluated in this project — Temperature Scaling (TS), Ensemble Temperature Scaling (ETS), Top-versus-All (TvA), and Density-Aware Calibration (DAC) — when the test distribution **shifts or drifts** from the training distribution.

The central finding is:

> **All post-hoc methods are working around a model that was never trained to be uncertain. There is a ceiling to how much you can fix at calibration time.**

---

##  Experimental Setup

| Component | Description |
|-----------|-------------|
| **Model** | DenseNet-BC-40, growth rate `k=12`, trained from scratch |
| **Dataset** | CIFAR-100 (100 classes, 50k train / 10k test, 32×32 px) |
| **Corruptions** | 13 types × 5 severity levels (CIFAR-100-C benchmark) |
| **Calibration split** | Fixed 5,000-sample validation set, saved by index |
| **Metrics** | ECE (15 fixed bins), Adaptive ECE (equal-mass bins), NLL |

---

##  Corruption Categories

The 13 corruption types are grouped into four families:

| Category | Corruptions |
|----------|-------------|
| **Noise** | Gaussian, Shot, Impulse |
| **Blur** | Defocus, Glass, Motion, Zoom |
| **Weather** | Fog, Brightness, Contrast |
| **Digital** | Elastic Transform, Pixelate, JPEG Compression |

Each is applied at **severity levels 1–5**, where level 5 represents the most extreme distribution shift.

---

##  Limitation 1 — Temperature Scaling Fails Under Shift

**Root Cause:** TS fits a single global scalar `T` on clean i.i.d. validation data. It has **no mechanism to adapt** when the test distribution changes.

**Observed Behavior:**
- Clean ECE: `0.135` — reasonable calibration on i.i.d. data
- Mean ECE under shift: `0.456` — severe degradation
- Gaussian noise @ severity 5: ECE = `0.778`, NLL = `28.9`

> A completely uniform predictor over 100 classes achieves NLL = `4.61`.  
> TS at worst noise conditions scores NLL = `28.9` — **6× worse than random**.

**Mechanism:** At high corruption severity, model accuracy collapses to near-chance (~1.5%), but confidence stays high because the fixed temperature `T` was calibrated on a cleaner distribution. The logit magnitudes actually *increase* for corrupted inputs (producing extreme softmax values), which TS cannot correct.

**Affected Corruption Types (worst):**
- Gaussian noise, Shot noise, Impulse noise (all severity levels)
- Contrast at severity 5 (images become near-black)
- JPEG compression at severity 5 (fine texture destruction)

---

##  Limitation 2 — ETS Structural Floor, But Not Shift-Aware

**Root Cause:** ETS adds a uniform mixing weight `w₃` and a raw-softmax weight, expanding from 1 to 4 parameters. The uniform component provides a **structural ceiling on overconfidence**, but still fitted only on clean data.

**What ETS Does Right:**
- Clean ECE: `0.066` (vs `0.135` for TS — 51% improvement)
- Mean ECE under shift: `0.333` (vs `0.456` for TS — 27% improvement)
- Mean NLL under shift: `4.29` (vs `8.65` for TS — halved)
- Gaussian noise @ severity 5: NLL = `6.5` (vs TS: `28.9`)

**Remaining Limitation:**
- ETS does **not detect corruption** — it just structurally cannot be *as wrong* as TS.
- On fog at severity 5: ECE = `0.236` — still significantly miscalibrated.
- On noise corruptions, ETS degrades but stays bounded; for weather/digital corruptions the improvement is most visible.
- ETS was fitted on clean data and has no access to any shifted samples at inference time.

---

##  Limitation 3 — TvA Paradox: Near-Zero ECE ≠ Good Calibration

**Root Cause:** TvA trains `K=100` per-class logistic scalers (`aₖ, bₖ`) on 5,000 clean validation samples. The 200 parameters **overfit the clean distribution**, creating a two-errors-cancelling artefact under shift.

**What Appears to Happen:**
- Under heavy corruption (e.g., Gaussian noise @ severity 5): ECE = `0.010` — seemingly excellent
- Under pixelate @ severity 5: ECE = `0.003`

**Why This Is an Artefact:**
- TvA aggressively pushes confidence **down** (because it corrects for overconfidence on clean data).
- Under severe corruption, the model's *accuracy* also collapses to ~1.4% (≈ 1/100 random).
- When both confidence and accuracy converge to `1/K ≈ 0.01`, ECE is near zero **by construction** — not genuine calibration.
- TvA's NLL at Gaussian noise severity 5: `5.11` (only slightly above uniform = `4.61`) — the model has **given up on prediction entirely**.

**Hard Evidence Against TvA:**
- Clean accuracy: `0.499` (vs `0.653` for TS/ETS) — **15% accuracy loss on clean data**, caused by miscalibrated per-class scalers disrupting the argmax.
- Elastic transform @ severity 5: ECE = `0.421` — worst of any method. Here the image statistics barely change, so TvA's aggressive downscaling fires at full strength without accuracy collapsing to compensate.

> **Rule:** Always check NLL alongside ECE when evaluating calibration under shift. ECE alone is insufficient and can be misleading.

---

##  Limitation 4 — DAC: Right Idea, Imprecise Proxy

**Root Cause:** DAC adapts the temperature **per sample** based on the L2 distance in logit space between the test sample and the validation mean (`dx`). This is a crude Out-of-Distribution (OOD) signal.

**What DAC Does:**
- Clean ECE: `0.101` (vs TS: `0.135`)
- Mean ECE under shift: `0.409` (vs TS: `0.456`)
- Improvements are consistent but modest across all corruption types.

**Why the Proxy Fails:**
- **Moderate corruptions** (brightness, elastic transform, zoom blur): `dx` is small → adaptive temperature stays near `Tbase` → near-identical behavior to TS → ETS already outperforms both.
- **Severe noise**: Corrupted images produce **high-magnitude logits** (not low-magnitude), so `dx` grows but the logit *explosion* outpaces the adaptive temperature correction.
- NLL trajectory for DAC on Gaussian noise: `7.4 → 9.6 → 12.9 → 16.6 → 20.0` across severities 1–5. The temperature is growing too slowly relative to the logit explosion.

**What Would Fix It:**
- Mahalanobis distance in penultimate feature space
- kNN density estimation in representation space
- These provide a more reliable OOD signal than raw logit-space L2 distance.

---

## 📊 Summary Table: Methods Under Shift

| Method | Clean ECE ↓ | Clean Acc ↑ | Mean ECE (shift) ↓ | Mean NLL (shift) ↓ | ECE @ Severity 5 ↓ |
|--------|------------|------------|-------------------|-------------------|-------------------|
| **TS** | 0.135 | 0.653 | 0.456 | 8.65 | 0.533 |
| **ETS** | 0.066 | 0.653 | 0.333 | 4.29 | 0.403 |
| **TvA** | 0.474 | 0.499 | 0.142 *(artefact)* | 4.30 | 0.088 *(artefact)* |
| **DAC** | 0.101 | 0.653 | 0.409 | — | — |

>  TvA metrics marked as artefact are **not genuine calibration** — see Limitation 3.

---

### Corruption-Specific Failure Modes

### Noise (Gaussian / Shot / Impulse) — *Most Severe*
- TS and DAC **collapse**: extremely high confidence on pure-noise images.
- ETS **degrades but stays bounded** due to the uniform mixing floor.
- TvA achieves near-zero ECE via the cancellation mechanism (not genuine robustness).

### Blur (Defocus / Glass / Motion / Zoom) — *Moderate*
- Zoom blur is most revealing: preserves image structure while zooming in.
- All methods except TvA perform reasonably — logit magnitudes shrink naturally.
- TS ECE @ zoom severity 5: `0.340` (much better than `0.778` on Gaussian noise).
- ETS consistently leads.

### Weather (Fog / Brightness / Contrast) — *Mixed*
- Fog and Brightness: ETS is most clearly better than TS (ECE `0.236` vs `0.372` on fog @ severity 5).
- Contrast @ severity 5: **catastrophic for all methods** — images become near-black (structureless).

### Digital (Elastic / Pixelate / JPEG) — *Mixed*
- Elastic transform: near-benign — all methods maintain near-clean performance.
- JPEG and Pixelate: significant degradation. ETS maintains the best ECE.
- Pixelate: NLL plateaus for TS and DAC at severity 3–5, indicating uniform wrongness.

---

##  Paths to Improvement

| Limitation | Proposed Fix |
|-----------|-------------|
| **TS/ETS not shift-aware** | Train with label smoothing, mixup, or CutMix for intrinsically better-calibrated base models |
| **DAC crude OOD proxy** | Replace L2 logit distance with Mahalanobis distance in penultimate feature space, or kNN density estimation |
| **TvA overfitting** | Use ≥20,000 validation samples; the 200 per-class parameters need substantially more data to generalize |
| **All methods: fixed at calibration time** | Explore test-time adaptation (e.g., TTT, tent, MEMO) that updates calibration parameters online |
| **DenseNet-BC-40 inherently overconfident** | Use larger, better-regularized models; modern training techniques produce models that are less overconfident to begin with |

---

##  Metric Guidance

When evaluating calibration under distribution shift, **do not rely on ECE alone**:

| Metric | What It Measures | Limitation |
|--------|-----------------|-----------|
| **ECE (fixed bins)** | Weighted average gap between confidence and accuracy | Skewed when confidence values cluster; TvA exploits this |
| **Adaptive ECE** | Same as ECE but with equal-mass bins | More reliable for non-standard confidence distributions |
| **NLL** | Proper scoring rule — penalizes both wrong predictions and overconfidence | Cannot be gamed by confidence-collapsing artefacts |

> **Best practice:** Report all three. A method is only genuinely calibrated under shift if all three metrics improve simultaneously.

---

##  Related Files

| File | Role |
|------|------|
| `evaluate_shift.py` | Full benchmark loop: 4 methods × 13 corruptions × 5 severities |
| `temperature_scaling.py` | All calibrators + ECE/NLL metrics |
| `corruptions.py` | 13 image corruption types, 5 severity levels each |
| `plot_per_corruption.py` | Per-corruption ECE and NLL visualization utilities |
| `summary.ipynb` | Full reproduction notebook with analysis and figures |

---

##  References

- Guo et al. (2017). *On Calibration of Modern Neural Networks.* ICML.
- Ovadia et al. (2019). *Can You Trust Your Model's Uncertainty?* NeurIPS.
- Rahimi et al. (2020). *Intra Order-Preserving Functions for Calibration of Multi-Class Neural Networks.* (ETS)
- Tomani et al. (2024). *Towards Trustworthy Predictions from Deep Neural Networks with Fast Adversarial Calibration.* (TvA)
- Tomani et al. (2023). *Post-Hoc Uncertainty Calibration for Domain Drift Scenarios.* (DAC)


