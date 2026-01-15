import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import re

def plot_curves(log_dir, task_name="Node Classification"):
    files = glob.glob(os.path.join(log_dir, "*_loss_train.npy"))
    
    for f in files:
        base = os.path.basename(f).replace("_loss_train.npy", "")
        loss_train = np.load(f)
        loss_val_path = f.replace("loss_train", "loss_val")
        
        plt.figure()
        plt.plot(loss_train, label='Train Loss')
        if os.path.exists(loss_val_path):
            loss_val = np.load(loss_val_path)
            plt.plot(loss_val, label='Val Loss')
            
        plt.title(f'{task_name} Loss: {base}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(log_dir, f"{base}_loss.png"))
        plt.close()
        
def plot_lp_curves(log_dir):
    files = glob.glob(os.path.join(log_dir, "*_lp_loss.npy"))
    for f in files:
        base = os.path.basename(f).replace("_lp_loss.npy", "")
        loss = np.load(f)
        
        plt.figure()
        plt.plot(loss, label='Train Loss')
        plt.title(f'Link Prediction Loss: {base}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(log_dir, f"{base}_loss.png"))
        plt.close()

def parse_results(log_dir):
    results = {}
    res_file = os.path.join(log_dir, "results.txt")
    if not os.path.exists(res_file):
        return results
        
    with open(res_file, 'r') as f:
        for line in f:
            # Format: cora_l2_n0_d0.5_de0.0 Test Acc: 0.81
            parts = line.strip().split()
            if "Test Acc:" in line:
                key = parts[0]
                val = float(parts[-1])
                results[key] = val
    return results

def plot_ablation(results, param_name, search_pattern):
    # Filter results matching search pattern
    # search_pattern e.g. "cora_l" to match layers
    filtered = {}
    for k, v in results.items():
        if "noself" in k and param_name != "SelfLoops": continue
        
        # Regex to extract param value
        # Pattern examples: 
        # layers: _l(\d+)_
        # dropedge: _de([\d\.]+)_
        # pairnorm: _n(\d+)_
        
        match = re.search(search_pattern, k)
        if match:
            param_val = match.group(1)
            # Group by dataset? assume cora for now or filter by dataset in key
            if "cora" in k:
                filtered[param_val] = v
                
    if not filtered:
        return

    # Sort
    sorted_keys = sorted(filtered.keys(), key=lambda x: float(x))
    vals = [filtered[k] for k in sorted_keys]
    
    plt.figure()
    plt.bar(sorted_keys, vals)
    plt.title(f'Effect of {param_name} (Cora)')
    plt.xlabel(param_name)
    plt.ylabel('Test Accuracy')
    plt.ylim(0, 1.0)
    for i, v in enumerate(vals):
        plt.text(i, v + 0.01, str(round(v, 4)), ha='center')
    plt.savefig(os.path.join("plots", f"ablation_{param_name}.png"))
    plt.close()

if __name__ == "__main__":
    plot_curves("plots", "Node Classification")
    plot_lp_curves("plots_lp")
    
    results = parse_results("plots")
    
    # 1. Self Loops (Bar chart: With vs Without)
    # Keys: cora_l2_n0_d0.5_de0.0 (default) vs ..._noself
    cora_default = results.get("cora_l2_n0_d0.5_de0.0", 0)
    cora_noself = results.get("cora_l2_n0_d0.5_de0.0_noself", 0)
    
    if cora_default > 0 or cora_noself > 0:
        plt.figure()
        plt.bar(["With Self-Loop", "No Self-Loop"], [cora_default, cora_noself])
        plt.title("Effect of Self-Loops (Cora)")
        plt.ylabel("Test Accuracy")
        plt.savefig(os.path.join("plots", "ablation_selfloop.png"))
        plt.close()

    # 2. Layers
    plot_ablation(results, "Layers", r"_l(\d+)_")
    
    # 3. DropEdge
    plot_ablation(results, "DropEdge", r"_de([\d\.]+)") # Note: no trailing underscore potentially if at end? My naming is ..._de0.0
    # wait, naming is ..._de0.0 or ..._de0.0_noself
    # Regex needs to be careful.
    
    # 4. PairNorm
    # Compare cora_l5_n1_... vs cora_l5_n0_...
    # Lets just find them explicitly
    # Naming: ..._n0_... is no PairNorm, ..._n1_... is PairNorm
    # We ran it for layers=5
    # Find keys with l5_n1 and l5_n0
    cora_deep_pn = 0
    cora_deep_base = 0
    for k, v in results.items():
        if "cora_l5" in k:
            if "_n1_" in k:
                cora_deep_pn = v
            elif "_n0_" in k:
                cora_deep_base = v
    
    if cora_deep_pn > 0 or cora_deep_base > 0:
        plt.figure()
        plt.bar(["Baseline (5 Layers)", "PairNorm (5 Layers)"], [cora_deep_base, cora_deep_pn])
        plt.title("Effect of PairNorm on Deep GCN (Cora)")
        plt.ylabel("Test Accuracy")
        plt.savefig(os.path.join("plots", "ablation_pairnorm.png"))
        plt.close()
