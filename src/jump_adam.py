import torch
from torch.optim import Optimizer

class JumpAdam(Optimizer):
    """
    JumpAdam: A Geometry-Aware Optimizer.
    
    Combines standard Adam (for safe traversal of rough terrain) with a 
    Newton-style Secant method (for jumping to the bottom of smooth basins).
    
    The optimizer assumes the loss landscape is locally composed of Distance 
    basins. It estimates the distance to the minimum by comparing the change 
    in gradient relative to the change in position (Curvature).
    
    Hyperparameters:
        params (iterable): Iterable of parameters to optimize.
        lr (float): Base learning rate for the Adam fallback.
        betas (Tuple[float, float]): Coefficients for computing running averages of gradient and its square.
        eps (float): Term added to the denominator to improve numerical stability.
        weight_decay (float): Weight decay (L2 penalty).
        trust_coeff (float): The Trust Region coefficient. If the Jump step is 
                             larger than trust_coeff * Adam_Step, we reject it 
                             and fall back to Adam.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, trust_coeff=5.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        trust_coeff=trust_coeff)
        super(JumpAdam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []
            
            # Extract state for this group
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('JumpAdam does not support sparse gradients')
                    grads.append(p.grad)

                    state = self.state[p]

                    # Lazy State Initialization
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # History for Secant Method
                        state['prev_param'] = p.clone().detach()
                        state['prev_grad'] = p.grad.clone().detach()

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    state_steps.append(state['step'])

            # Apply Jump Adam Logic
            self._jump_adam_update(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps
            )

        return loss

    def _jump_adam_update(self, group, params, grads, exp_avgs, exp_avg_sqs, state_steps):
        beta1, beta2 = group['betas']
        lr = group['lr']
        eps = group['eps']
        trust_coeff = group['trust_coeff']
        weight_decay = group['weight_decay']

        for i, param in enumerate(params):
            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            state = self.state[param]
            
            # Increment step
            state['step'] += 1
            step = state['step']

            # Weight Decay
            if weight_decay != 0:
                grad = grad.add(param, alpha=weight_decay)

            # --- 1. Calculate Adam Candidate (The Safe Path) ---
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            
            denom = (exp_avg_sq.sqrt() / torch.sqrt(torch.tensor(bias_correction2))) + eps
            adam_step_size = lr / bias_correction1
            
            # The actual step vector Adam wants to take
            adam_step = -adam_step_size * (exp_avg / denom)

            # --- 2. Calculate Jump Candidate (The Fast Path) ---
            # Retrieve previous state
            prev_param = state['prev_param']
            prev_grad = state['prev_grad']

            # Secant Deltas
            delta_p = param - prev_param
            delta_g = grad - prev_grad
            
            # Calculate Curvature (Slope of the Gradient)
            # k = delta_g / delta_p
            # We want jump = -grad / k = -grad * (delta_p / delta_g)
            
            # Avoid division by zero in curvature calculation
            # If delta_g is 0 (gradient didn't change), curvature is 0 (flat). 
            # We cannot jump on a flat plane without solving linear system. Fallback.
            curvature_mask = delta_g.abs() > eps
            
            # We initialize jump_step with zeros (which implies "do nothing" -> "fallback")
            jump_step = torch.zeros_like(param)
            
            # Only compute jump where geometry is valid
            if curvature_mask.any():
                # Element-wise jump calculation
                # jump = -grad_current * (delta_p / delta_g)
                # We use 'where' to handle the safe masking
                safe_delta_g = torch.where(curvature_mask, delta_g, torch.ones_like(delta_g))
                
                raw_jump = -grad * (delta_p / safe_delta_g)
                
                # --- 3. The Gating Mechanism (Trust Region) ---
                
                # Condition A: Convexity Check
                # If delta_g and delta_p have different signs, curvature is negative (concave).
                # Wait, Secant Curvature k = dg/dp.
                # If we are in a basin, higher p should mean higher g (positive k).
                # Actually, strictly, k > 0 is required for a minimum.
                # k = delta_g / delta_p. 
                # We check if (delta_g * delta_p) > 0.
                convex_mask = (delta_g * delta_p) > 0
                
                # Condition B: Trust Region
                # Is the Jump Step insane? Compare magnitude to Adam Step.
                # We do this element-wise.
                adam_mag = adam_step.abs()
                jump_mag = raw_jump.abs()
                trust_mask = jump_mag <= (trust_coeff * adam_mag)
                
                # Final Valid Mask: Curvature Valid AND Convex AND Trusted
                valid_jump_mask = curvature_mask & convex_mask & trust_mask
                
                # Select: Jump where valid, Adam where invalid
                final_update = torch.where(valid_jump_mask, raw_jump, adam_step)
            else:
                # If no parameters have valid curvature info, full fallback
                final_update = adam_step

            # --- 4. Update State for Next Step ---
            # We must store the CURRENT parameter and gradient before we modify it
            state['prev_param'].copy_(param)
            state['prev_grad'].copy_(grad)

            # --- 5. Apply Update ---
            param.add_(final_update)