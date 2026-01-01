# Jump Adam: A Geometry-Aware Optimizer

## 1. Abstract

**Jump Adam** is an experimental optimization algorithm derived from the theory that Neural Networks function as **Distance Estimators**.

Current optimizers (SGD, Adam) operate on the assumption of **Hill Climbing**: blindly following the local gradient slope. Jump Adam operates on the assumption of **Triangulation**: using local geometry (slope, curvature, and loss magnitude) to estimate the exact distance to the solution and "teleport" there in a single update.

## 2. Theoretical Basis

### 2.1 The Distance Hypothesis

In standard Deep Learning, Loss is treated as a scalar to be minimized. In our framework, Loss () is a proxy for the geometric distance () to the optimal parameters .


* **Linear Basins (MAE/ReLU):** . The loss value *is* the distance.
* **Quadratic Basins (MSE):** . The gradient change (curvature) defines the distance.

### 2.2 The "Slow Compiler" Problem

Standard Gradient Descent is a "fuzzer": it tweaks weights and checks if the error dropped. It is effective but inefficient.
Once a network enters a **Basin of Attraction** (stable active ReLU configuration), the geometry simplifies to a smooth convex shape. In this phase, iterative crawling is unnecessary. If we can identify the basin type (Linear vs. Quadratic), we can solve for the minimum analytically.

## 3. The Algorithm

Jump Adam implements a **Hybrid Solver** that switches strategies based on local geometry.

### 3.1 Mode A: Quadratic Jump (The Secant Method)

*Used when curvature is detected (e.g., MSE Loss).*
By monitoring the change in gradient relative to the change in parameters, we estimate the basin's "stiffness" () and jump to the bottom.


### 3.2 Mode B: Linear Jump (Loss Projection)

*Used when curvature is zero (e.g., MAE Loss, Flat ReLU Mesas).*
When gradients are constant (), the Secant method fails (divide by zero). However, in these regions, the Loss value itself is proportional to distance. We project the current Loss magnitude along the gradient direction.


### 3.3 Safety Mechanisms (The Gating Logic)

To prevent instability when the "Basin Assumption" is violated:

1. **Consistency Check:** Only jump if gradients are stable (no oscillating).
2. **Trust Region:** If the calculated "Jump" is massively larger than a standard Adam step (defined by `trust_coefficient`), we reject it as a hallucination.
3. **Fallback:** If neither Jump mode is confident, the optimizer defaults to standard **Adam** behavior.

## 4. Implementation Plan

We implement `JumpAdam` as a custom PyTorch Optimizer with state-tracking for gradients and parameters.

**Key State Variables:**

* `prev_param`: 
* `prev_grad`: 
* `exp_avg/sq`: Standard Adam momentum buffers.

**Hyperparameters:**

* `lr`: Base learning rate (for Adam fallback).
* `jump_lr`: Scaling factor for jumps (usually 1.0 for perfect projection).
* `trust_coeff`: Max ratio of (Jump Step / Adam Step) allowed.
* `linear_threshold`: Curvature value below which we switch to Linear Mode.

## 5. Expected Behavior

* **Polyhedral/Linear Basins:** The optimizer detects zero curvature and uses "Loss Projection" to snap to the manifold (solving `Abs` or `ReLU` constraints instantly).
* **Quadratic Basins:** The optimizer detects curvature and uses "Secant Jump" to hit the vertex of the parabola.
* **Complex/Rough Terrain:** The optimizer falls back to Adam, crawling safely until a basin is found.