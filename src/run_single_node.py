import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from jump_adam import JumpAdam

def run_jump_vs_crawl():
    print("=== The 'See Adam Jump' Experiment ===")
    
    # 1. Setup the Problem: y = |x - 10|
    # A perfect V-shape basin. Minimum is at x = 10.
    # We start at x = 0.
    target = 10.0
    start_val = 0.0
    
    # Model 1: The Crawler (Standard Adam)
    x_crawl = nn.Parameter(torch.tensor([start_val]))
    opt_crawl = torch.optim.Adam([x_crawl], lr=1.0) # Aggressive LR
    
    # Model 2: The Jumper (Jump Adam)
    x_jump = nn.Parameter(torch.tensor([start_val]))
    opt_jump = JumpAdam([x_jump], lr=1.0, trust_coeff=10.0)
    
    history_crawl = []
    history_jump = []
    
    print(f"Goal: Reach {target} from {start_val}")
    
    # 2. The Race (10 Steps)
    for t in range(10):
        # --- Crawler Step ---
        opt_crawl.zero_grad()
        loss_crawl = torch.abs(x_crawl - target)
        loss_crawl.backward()
        opt_crawl.step()
        history_crawl.append(x_crawl.item())
        
        # --- Jumper Step ---
        opt_jump.zero_grad()
        loss_jump = torch.abs(x_jump - target)
        loss_jump.backward()
        opt_jump.step()
        history_jump.append(x_jump.item())
        
        print(f"Step {t+1}: Adam={x_crawl.item():.4f} | Jump={x_jump.item():.4f}")

    # 3. The Result
    final_err_crawl = abs(x_crawl.item() - target)
    final_err_jump = abs(x_jump.item() - target)
    
    print("\n=== Results ===")
    print(f"Adam Error: {final_err_crawl:.6f}")
    print(f"Jump Error: {final_err_jump:.6f}")
    
    if final_err_jump < 1e-5 and final_err_crawl > 0.1:
        print("\nSUCCESS: Adam walked, but Jump teleported.")
    else:
        print("\nFAILURE: Jump didn't stick the landing.")

if __name__ == "__main__":
    run_jump_vs_crawl()