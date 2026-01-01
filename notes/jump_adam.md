# Jump Adam: Algorithm Specification

## 1. Core Logic

The optimizer combines two distinct update mechanisms:

1. **The Base (Adam):** Provides a robust, safe default step based on momentum and RMS scaling.
2. **The Accelerator (Jump):** Provides an opportunistic, aggressive step based on local curvature estimation (Secant Method).

The optimizer calculates both candidates at every step and uses a **Gating Function** to decide whether to "Walk" (Adam) or "Teleport" (Jump).

## 2. State Variables (Per Parameter)

For each parameter tensor , we maintain:

* `exp_avg` (): Exponential moving average of gradient (Adam).
* `exp_avg_sq` (): Exponential moving average of squared gradient (Adam).
* `prev_param` (): Parameter value from the previous step.
* `prev_grad` (): Gradient value from the previous step.

## 3. The Update Cycle (Step )

### Step 3.1: Calculate Adam Candidate

Standard Adam implementation to get the "Safe Step".


### Step 3.2: Estimate Curvature (Secant Method)

Calculate the change in position and gradient relative to the previous step.


Estimate the diagonal Hessian (curvature) :


### Step 3.3: Calculate Jump Candidate

If the local geometry is a convex basin, the distance to the minimum is .


*Note: This simplifies to *

### Step 3.4: The Gating Function (Trust Region)

We only execute the Jump if the geometry is "Well-Behaved."

**Rejection Conditions (Fallback to Adam):**

1. **Non-Convexity:** If . (The surface is flat or curving downwards; the secant method would project us to infinity or backwards).
2. **Instability:** If  or  (No movement info).
3. **Trust Violation:** If .
*  is the `trust_coefficient` (e.g., 5.0).
* If the Jump wants to move 100x further than Adam, it's likely a hallucination caused by a noisy local gradient. We clamp it or reject it.



**Update Rule:**
If (Valid Curvature) AND (Inside Trust Region):



Else:


### Step 3.5: State Maintenance

Store current values for the next iteration.


## 4. Hyperparameters

| Parameter | Symbol | Default | Description |
| --- | --- | --- | --- |
| `lr` |  | 1e-3 | Base learning rate for Adam fallback. |
| `betas` |  | (0.9, 0.999) | Adam momentum coefficients. |
| `trust_coeff` |  | 5.0 | Max multiplier allowing Jump to exceed Adam. |
| `eps` |  | 1e-8 | Numerical stability term. |