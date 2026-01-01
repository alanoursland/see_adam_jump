# Jump Adam: Algorithm Specification

## 1. Core Logic

The optimizer operates as a **Hybrid Switch**, selecting the best update strategy based on the local geometry of the loss landscape:

1. **The Base (Adam):** A robust fallback for rough terrain, using momentum and RMS scaling.
2. **Mode A (Quadratic Jump):** A Newton-style Secant step for basins with detectable curvature (e.g., MSE, smooth valleys).
3. **Mode B (Linear Jump):** A "Loss Projection" step for basins with zero curvature (e.g., L1/MAE, flat ReLU mesas), where the loss value itself acts as the distance map.

The optimizer calculates these candidates at every step and uses a **Gating Function** to decide whether to Crawl (Adam), Jump (Secant), or Project (Linear).

## 2. State Variables (Per Parameter)

For each parameter tensor , we maintain:

* `exp_avg` (): Exponential moving average of gradient (Adam).
* `exp_avg_sq` (): Exponential moving average of squared gradient (Adam).
* `prev_param` (): Parameter value from the previous step.
* `prev_grad` (): Gradient value from the previous step.

## 3. The Update Cycle (Step )

### Step 3.1: Calculate Adam Candidate

Standard Adam implementation to generate the "Safe Step" ().


### Step 3.2: Analyze Geometry (Secant Method)

Calculate the physical changes since the last step to estimate the local topology.


Estimate the diagonal curvature (stiffness) :


### Step 3.3: Calculate Jump Candidate

We select the Jump strategy based on the detected curvature .

**Case A: Quadratic Basin ()**
If the gradient is changing, we use the Secant method to find the vertex of the parabola.


**Case B: Linear Basin ()**
If the gradient is constant (flat slope), curvature is zero. We cannot use the Secant method.
Instead, we use the **Loss Value** () as a direct proxy for distance. We project the Loss magnitude along the gradient direction.



*(Note: This projects the scalar Loss distance onto the vector space).*

### Step 3.4: The Gating Function (Trust Region)

We filter the candidate Jump step  through safety checks.

**Rejection Conditions (Fallback to Adam):**

1. **Bad Geometry:**
* If using **Quadratic Mode**: Reject if  (Non-convex/Hill).
* If using **Linear Mode**: Reject if  is not provided (Closure missing).


2. **Instability:** If  (Stalled) or  (No curvature info *and* no Loss info).
3. **Trust Violation:** If .
*  is the `trust_coeff`.
* Prevents massive overshooting if the local geometry estimate is noisy.



**Update Rule:**


### Step 3.5: State Maintenance

Store current values for the next iteration.


## 4. Hyperparameters

| Parameter | Symbol | Default | Description |
| --- | --- | --- | --- |
| `lr` |  | 1e-3 | Base learning rate for Adam fallback. |
| `betas` |  | (0.9, 0.999) | Adam momentum coefficients. |
| `trust_coeff` |  | 5.0 | Max multiplier allowing Jump to exceed Adam. |
| `linear_threshold` |  | 1e-6 | Curvature threshold below which we switch to Linear (Loss) Mode. |
| `eps` |  | 1e-8 | Numerical stability term. |