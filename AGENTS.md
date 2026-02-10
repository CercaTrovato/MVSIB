# AGENTS.md

## 1. Project Purpose

This repository implements **MVSIB**, a multi-view clustering framework with:

* Multi-view autoencoders + consensus fusion (`network.py`)
* Pseudo-label and clustering center updates (`model.py`)
* Uncertainty estimation + curriculum gating
* FN/HN (False Negative / Hard Negative) discrimination
* Multi-level contrastive objectives (`loss.py`)
* Warm-up reconstruction pretraining + contrastive main training (`train.py`)

**Primary goal of any modification**:

> Improve the model, training strategy, loss design, or experimental protocol **without breaking reproducibility or baselines**.

This is a **research codebase**, not a product codebase. Stability, traceability, and comparability are higher priority than aggressive refactoring.

---

## 2. Core Principles (Must Follow)

1. **Reproducibility first**

   * Do not change default seeds, dataset splits, or evaluation protocols unless explicitly requested.
   * Any change affecting results must be clearly documented in code comments.

2. **Minimal, localized modifications**

   * Prefer modifying **one module at a time** (e.g., `loss.py` or `model.py`) instead of cross-cutting refactors.
   * Do NOT restructure the whole project unless explicitly instructed.

3. **Baseline safety**

   * Do NOT remove or overwrite:

     * Existing losses
     * FN/HN mechanism
     * Uncertainty estimation
     * Warm-up pretraining
   * If adding new components, keep the old ones runnable via flags or config switches.

4. **Experiment traceability**

   * New methods should be:

     * Toggleable by flags or parameters
     * Easy to ablate or disable
   * Keep old behavior as a reference path.

---

## 3. Directory & File Responsibilities

* `train.py`

  * Entry point: argument parsing, training loop, logging, evaluation
  * ❗ Do not hard-code experimental tricks here unless strictly necessary.

* `test.py`

  * Evaluation-only script
  * ⚠ Keep consistent with `train.py` model signatures.

* `dataloader.py`

  * Multi-view dataset loading and normalization
  * ❗ Do not silently change preprocessing without explicit instruction.

* `network.py`

  * Model architecture:

    * Multi-view encoders/decoders
    * Fusion module
    * Cluster centers
    * Uncertainty head
    * FN/HN classifier
  * ✅ Architecture-level innovations go here.

* `model.py`

  * Core algorithm:

    * kNN graph construction
    * Pseudo-label update
    * Warm-up training
    * Contrastive training logic
    * Gating, FN/HN handling, uncertainty logic
  * ✅ Training strategy changes go here.

* `loss.py`

  * All losses:

    * Feature contrastive
    * Cluster-level InfoNCE
    * Cross-view weighted InfoNCE
    * GMM / MMD / repulsion / boundary constraints
    * Uncertainty regression
    * Reconstruction loss
  * ✅ New loss terms should be added here, not scattered elsewhere.

* `metric.py`

  * Evaluation metrics: ACC / NMI / ARI / PUR / F-score
  * ❗ Do not change metric definitions unless explicitly requested.

* `logger.py`

  * Logging utilities
  * Keep logging consistent and comparable across experiments.

---

## 4. Allowed Modification Types

You MAY:

* Add new loss terms (in `loss.py`) with:

  * Clear naming
  * Clear weighting hyperparameters
  * Optional on/off switch

* Modify or extend:

  * Uncertainty estimation strategy
  * FN/HN discrimination features or classifier
  * Gating / curriculum schedule
  * Fusion strategy or latent structure
  * Contrastive objectives (cluster-level, feature-level, cross-view)

* Add:

  * Ablation options
  * Debug logging
  * Extra evaluation outputs

---

## 5. Forbidden or High-Risk Actions (Unless Explicitly Asked)

* ❌ Do NOT:

  * Rewrite the whole training loop
  * Remove existing losses or mechanisms
  * Change dataset preprocessing silently
  * Change evaluation metrics or their computation
  * Rename files or reorganize folders
  * Break compatibility with existing checkpoints

* ⚠ Avoid:

  * “Clean-up” refactors that change behavior
  * Merging multiple conceptual changes in one modification
  * Introducing new dependencies without necessity

---

## 6. Coding & Research Style Rules

* Keep variable names **consistent with the paper and existing code**:

  * `u`, `u_hat`, `common_z`, `q`, `p`, `centers`, `sigma`, `gate`, etc.

* Any new component must include:

  * A short inline comment explaining **what it does conceptually**
  * A note on **which part of the method it corresponds to**

* Prefer:

  * Small, reviewable diffs
  * One idea per change

---

## 7. Experiment Safety Checklist (Before Finishing Any Change)

Before finalizing any modification, ensure:

* [ ] Code runs with the original settings
* [ ] Original behavior can still be reproduced
* [ ] New behavior is controlled by flags or parameters
* [ ] Logs clearly indicate when the new method is active
* [ ] Metrics pipeline is unchanged

---

## 8. How the Agent Should Think

When modifying this repository, always ask:

1. Does this preserve the **original method as a baseline**?
2. Can this be **ablated or disabled** easily?
3. Does this **break reproducibility or comparability**?
4. Is this change **clearly attributable to one research idea**?

If the answer to any is “no”, stop and revise the approach.

---

## 9. Priority of Changes

1. Correctness & reproducibility
2. Research clarity & ablation friendliness
3. Experimental controllability
4. Code elegance (last priority)

---

## 10. Summary

This is a **research iteration codebase** for MVSIB.
The agent’s job is to:

> Safely, incrementally, and transparently improve the model and experiments — not to rewrite or “clean up” the system.
