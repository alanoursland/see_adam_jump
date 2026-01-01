# Jump Adam: A Geometry-Aware Optimizer

## 1. Abstract

**Jump Adam** is an experimental optimization algorithm derived from the theory that Neural Networks function as **Distance Estimators** rather than feature detectors.

Current optimizers (SGD, Adam) operate on the assumption of **Hill Climbing**: they look at the local slope (gradient) and take a small, cautious step in that direction, hoping not to overshoot.

Jump Adam operates on the assumption of **Triangulation**: it assumes the loss function  is a proxy for the geometric distance to the optimal solution . By monitoring the change in gradient relative to the change in position (Secant Method), it estimates the distance to the basin floor and attempts to "jump" directly to the solution in a single update step per parameter.

## 2. Theoretical Basis

### 2.1 The Distance Hypothesis

In standard Deep Learning theory, Loss is an arbitrary scalar value. In our framework, Loss is a measure of geometric misalignment:



Where  depends on the local topology (usually  for polyhedral/ReLU basins, or  for quadratic basins).

### 2.2 The "Slow Compiler" Problem

Standard Gradient Descent behaves like a "fuzzer":

1. Check slope.
2. Move slightly.
3. Re-evaluate.
4. Repeat 10,000 times.

This approach is necessary in the chaotic, non-convex initial phase of training. However, once the network enters a **Basin of Attraction** (a specific configuration of active ReLUs), the geometry becomes relatively smooth (often piecewise linear or quadratic).

In this "Basin Phase," crawling is inefficient. If we know the slope and the rate of change of the slope (curvature), we can calculate the intercept point (zero gradient) and teleport there.

### 2.3 The "One Node" Proof

Experiments with single-node models () demonstrate that if the learning rate is set to `lr=1` (effectively a full geometric projection), the model converges in **1 step**. Jump Adam attempts to generalize this behavior to deep networks by treating every parameter as an independent "One Node" problem.

## 3. The Algorithm

Jump Adam implements a **Diagonal Secant Method**. It approximates the curvature for each parameter independently using only first-order gradients from the current and previous steps.

### 3.1 The Estimator

For each parameter  at time step :

1. **Measure Motion:** 
2. **Measure Change in Gradient:** 
3. **Estimate Stiffness (Curvature):**



### 3.2 The Jump Rule

If we assume a local quadratic basin (or linear slope approaching a floor), the distance to the minimum is:


The "Jump" update is:



Substituting :


*Note: In a perfect quadratic bowl,  yields the exact solution.*

### 3.3 Safety Mechanisms (The Gating Logic)

Because the "Distance Hypothesis" only holds within a basin, blindly jumping can lead to instability (overshooting out of the basin). We apply **Trust Regions**:

1. **Consistency Check:** Only jump if the gradient sign is consistent (we are sliding down a wall, not oscillating in a valley).
2. **Curvature Sanity:** If  (flat) or  (non-convex), fallback to standard Adam/SGD.
3. **Clamp:** Limit the maximum "Jump" to a multiple of the standard Adam step size (e.g., ) to prevent exploding parameters.

## 4. Expected Behavior

* **Phase 1 (Chaos):** The algorithm behaves like standard Adam. Curvature estimates are noisy/negative, so the "Jump" gate stays closed.
* **Phase 2 (Basin Entry):** As gradients align, the algorithm detects stable curvature.
* **Phase 3 (The Warp):** The effective learning rate dynamically scales up (potentially to ), rapidly traversing the flat regions of the mesa and snapping to the minima.

## 5. Implementation Plan

We will implement `JumpAdam` as a custom PyTorch Optimizer.

**State Requirements:**

* `params`: 
* `prev_params`:  (Required to calculate )
* `prev_grads`:  (Required to calculate )
* `exp_avg`: Standard Adam momentum (for fallback).

**Key Hyperparameters:**

* `jump_lr`: Learning rate scaling factor for the jump (default 1.0).
* `trust_coefficient`: Max allowed ratio between Jump Step and Adam Step.