# Theory: Loss Topology and Distance Proxies

## 1. The Geometric Premise

Standard optimization theory treats the Loss Function  as an arbitrary scalar field to be minimized via local gradient descent. Our framework posits a stronger geometric constraint: **Loss is a direct proxy for the Euclidean distance to the optimal parameters .**

This relationship implies that the optimizer does not need to "search" for the bottom; it can **triangulate** the bottom if the mapping function  is known.

## 2. The Two Basin Regimes

Our empirical results identify two distinct geometric regimes based on the choice of Loss Function and the activation patterns of the network.

### 2.1 Quadratic Basins (The Parabola)

**Context:** Occurs when using **MSE Loss** or near the minimum of smooth convex functions where the loss surface is well-approximated by a quadratic.

**Proxy Relation:** Distance is proportional to the square root of the loss.


* **Evidence:** In experiments with `Abs1` and `ReLU1` models, the quantity  explains  of the variance in remaining epochs to convergence once the basin is reached.
* **Implication for Optimization:**
* The gradient magnitude scales linearly with distance: .
* The curvature (second derivative) is constant and non-zero.
* **Strategy:** The **Secant Method** is optimal here. By measuring the change in gradient () relative to step size (), we can estimate the curvature  and calculate the distance .



### 2.2 Linear Basins (The Cone)

**Context:** Occurs when using **L1/MAE Loss**, or in networks dominated by piecewise linear activations (`|Wx+b|` or `ReLU`) far from the minimum. In these models, the gradient structure is piecewise constant.

**Proxy Relation:** Distance is directly proportional to the loss.


* **Evidence:** We observed a linear relationship where  holds across multiple distance metrics due to norm equivalence in finite dimensions.
* **The Trap:** Standard second-order methods (Newton/Secant) fail here because they rely on curvature (). In a linear basin, gradients are constant, so  (zero curvature). If applied blindly, the Secant method detects "infinite distance" and stalls.
* **Strategy:** The **Loss Projection** method is required. Since we cannot infer distance from the *change* in slope (it's flat), we must use the *value* of the Loss itself to determine step size.



## 3. Basin Entry and Stability

The "Distance Proxy" hypothesis relies on the network being in a **Basin of Attraction**.

* **Chaotic Phase:** Early in training, gradients fluctuate as the network traverses non-convex ridges. Loss is not a reliable distance proxy here.
* **Basin Phase:** Once the network settles into a stable configuration, the local geometry becomes effectively convex. We detect this phase via **Stability of Geometry** cues:
1. **Gradient Stability:** The cosine similarity between successive updates stabilizes ().
2. **Distance Stability:** The change in parameter norms () drops below a threshold relative to the total norm.



Only when these stability conditions are met does `JumpAdam` engage its aggressive "Teleportation" modes.

## 4. Summary of Modes

| Regime | Loss Type | Geometry | Gradient () | Curvature () | Proxy Formula | Jump Strategy |
| --- | --- | --- | --- | --- | --- | --- |
| **Quadratic** | MSE | Parabola | Linear Decay | Constant  |  | **Secant:**  |
| **Linear** | MAE / ReLU | Cone / Ramp | Constant |  |  | **Projection:**  |

## 5. Conclusion

Optimizers must be "Loss-Aware." Relying solely on gradients (slope) discards the critical distance information encoded in the Loss magnitude. By identifying whether the local topology is **Linear** or **Quadratic** (via curvature estimation), `JumpAdam` can switch between **Projection** and **Secant** strategies to achieve near-instant convergence in stable basins.