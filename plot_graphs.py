## nalyze_lr_sweep.py
#
# Analyze LR sweep results to find critical LR and compare vendor stability windows.

import argparse
import pickle
from collections import defaultdict
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_results(result_file: str) -> Dict:
    """Load LR sweep results."""
    with open(result_file, "rb") as f:
        return pickle.load(f)


def analyze_stability(results: List[Dict]) -> Dict:
    """Analyze stability metrics per vendor/LR."""
    by_vendor_lr = defaultdict(list)
    
    for r in results:
        if "error" in r:
            continue
        key = (r["vendor"], r["lr"])
        by_vendor_lr[key].append(r)
    
    analysis = {}
    for (vendor, lr), runs in by_vendor_lr.items():
        diverged = sum(1 for r in runs if r.get("diverged", False))
        total = len(runs)
        final_losses = [r.get("final_loss", float('inf')) for r in runs if not r.get("diverged", False)]
        best_losses = [r.get("best_loss", float('inf')) for r in runs if not r.get("diverged", False)]
        
        analysis[(vendor, lr)] = {
            "diverged_count": diverged,
            "total_runs": total,
            "divergence_rate": diverged / total if total > 0 else 0.0,
            "avg_final_loss": np.mean(final_losses) if final_losses else float('inf'),
            "std_final_loss": np.std(final_losses) if len(final_losses) > 1 else 0.0,
            "avg_best_loss": np.mean(best_losses) if best_losses else float('inf'),
        }
    
    return analysis


def plot_lr_sweep(analysis: Dict, output_file: str, config: Dict):
    """Plot LR sweep results showing stability windows."""
    vendors = sorted(set(v for v, _ in analysis.keys()))
    lrs = sorted(set(lr for _, lr in analysis.keys()))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Divergence rate vs LR
    ax1 = axes[0]
    for vendor in vendors:
        divergence_rates = []
        vendor_lrs = []
        for lr in lrs:
            key = (vendor, lr)
            if key in analysis:
                divergence_rates.append(analysis[key]["divergence_rate"])
                vendor_lrs.append(lr)
        if vendor_lrs:
            ax1.plot(vendor_lrs, divergence_rates, marker='o', label=vendor, linewidth=2)
    
    ax1.set_xlabel("Learning Rate", fontsize=12)
    ax1.set_ylabel("Divergence Rate", fontsize=12)
    ax1.set_title("Divergence Rate vs Learning Rate", fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # Plot 2: Final loss vs LR (only for non-diverged runs)
    ax2 = axes[1]
    for vendor in vendors:
        final_losses = []
        vendor_lrs = []
        for lr in lrs:
            key = (vendor, lr)
            if key in analysis:
                avg_loss = analysis[key]["avg_final_loss"]
                if avg_loss < float('inf'):
                    final_losses.append(avg_loss)
                    vendor_lrs.append(lr)
        if vendor_lrs:
            ax2.plot(vendor_lrs, final_losses, marker='o', label=vendor, linewidth=2)
    
    ax2.set_xlabel("Learning Rate", fontsize=12)
    ax2.set_ylabel("Average Final Loss", fontsize=12)
    ax2.set_title("Final Loss vs Learning Rate (Non-Diverged)", fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_file}")


def find_critical_lr(analysis: Dict, vendor: str, lrs: List[float], threshold: float = 0.5) -> float:
    """Find the critical LR where divergence rate exceeds threshold."""
    for lr in sorted(lrs):
        key = (vendor, lr)
        if key in analysis:
            if analysis[key]["divergence_rate"] >= threshold:
                return lr
    return None


def detect_saturation(losses: List[float], window: int = 200, threshold: float = 0.01) -> int:
    """
    Detect when loss saturates (stops decreasing significantly).
    Returns the step index where saturation begins, or None if not saturated.
    
    Args:
        losses: List of loss values
        window: Number of steps to look back for saturation detection
        threshold: Relative change threshold (e.g., 0.01 = 1% change)
    """
    if len(losses) < window * 2:
        return None
    
    # Look for saturation in the last portion of training
    for i in range(len(losses) - window, window, -1):
        if i < window:
            break
        recent = losses[i:i+window]
        earlier = losses[i-window:i]
        
        if len(recent) == window and len(earlier) == window:
            recent_avg = np.mean(recent)
            earlier_avg = np.mean(earlier)
            
            # Check if loss has stopped decreasing significantly
            if earlier_avg > 0:
                relative_change = (earlier_avg - recent_avg) / earlier_avg
                if relative_change < threshold:
                    return i
    
    return None


def plot_loss_saturation_analysis(results: List[Dict], output_file: str, max_curves: int = 20):
    """
    Plot loss curves showing saturation points, with 2 rows:
    - Row 1: FP16 vendors (all LRs)
    - Row 2: BF16 vendors (all LRs)
    Shows different vendor implementations (base, trunc, rtn) and LRs.
    
    Args:
        results: List of result dictionaries from training
        output_file: Output path for the plot
        max_curves: Maximum number of curves to plot (to avoid clutter)
    """
    # Group results by vendor/LR, keeping only non-diverged runs
    by_vendor_lr = defaultdict(list)
        # Select representative runs (average across seeds for each vendor/LR)
    for r in results:
        if "error" in r or r.get("diverged", False):
            continue
        if "losses" not in r or not r["losses"]:
            continue
        key = (r["vendor"], r["lr"])
        by_vendor_lr[key].append(r)

    vendors = sorted(set(v for v, _ in by_vendor_lr.keys()))

    # Get unique LRs and vendors
    lrs = sorted(set(lr for _, lr in by_vendor_lr.keys()))


    # Create subplots: one per vendor, or combine if too many
    n_vendors = len(vendors)
    if n_vendors <= 4:
        fig, axes = plt.subplots(1, n_vendors, figsize=(5*n_vendors, 5))
        if n_vendors == 1:
            axes = [axes]
    else:
        # Too many vendors, create a grid
        n_cols = 3
        n_rows = (n_vendors + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
    
    for vendor_idx, vendor in enumerate(vendors):
        ax = axes[vendor_idx]

        for lr in lrs:
        
        # Plot loss curves for each FP16 vendor at this LR
        # for vendor in fp16_vendors:
            key = (vendor, lr)
            if key not in by_vendor_lr:
                continue
            
            runs = by_vendor_lr[key]
            if not runs:
                continue
            
            # Average losses across seeds
            max_steps = max(len(r["losses"]) for r in runs)
            avg_losses = np.zeros(max_steps)
            count = np.zeros(max_steps)
            
            for r in runs:
                losses = r["losses"]
                for i, loss in enumerate(losses):
                    if i < max_steps:
                        avg_losses[i] += loss
                        count[i] += 1
            
            # Compute average
            avg_losses = avg_losses / np.maximum(count, 1)
            steps = np.arange(1, len(avg_losses) + 1)

            # Plot curve
            label = f"LR={lr:.4f}"
            ax.plot(steps, avg_losses, label=label, linewidth=2, alpha=0.8)

            
            # Mark saturation point
            sat_step = detect_saturation(avg_losses.tolist())
            if sat_step is not None and sat_step < len(avg_losses):
                ax.axvline(x=sat_step, color='red', linestyle='--', alpha=0.5, linewidth=1)
                ax.plot(sat_step, avg_losses[sat_step], 'ro', markersize=8, alpha=0.7)
        
        ax.set_xlabel("Step", fontsize=11)
        ax.set_ylabel("Loss", fontsize=11)
        ax.set_title(f"{vendor}", fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

    for idx in range(len(vendors), len(axes)):
        axes[idx].axis('off') 

    plt.suptitle("Loss Saturation Analysis: Step vs Loss (Red markers indicate saturation)\n" + 
                 "(subscale=1e-4)", 
                 fontsize=14, fontweight='bold', y=1.02)   
 
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved saturation analysis plot: {output_file}")


def plot_loss_saturation(results: List[Dict], output_file: str, max_curves: int = 20):
    """
    Plot loss curves showing saturation points, with 2 rows:
    - Row 1: FP16 vendors (all LRs)
    - Row 2: BF16 vendors (all LRs)
    Shows different vendor implementations (base, trunc, rtn) and LRs.
    
    Args:
        results: List of result dictionaries from training
        output_file: Output path for the plot
        max_curves: Maximum number of curves to plot (to avoid clutter)
    """
    # Group results by vendor/LR, keeping only non-diverged runs
    by_vendor_lr = defaultdict(list)
    for r in results:
        if "error" in r or r.get("diverged", False):
            continue
        if "losses" not in r or not r["losses"]:
            continue
        key = (r["vendor"], r["lr"])
        by_vendor_lr[key].append(r)
    
    # Get unique LRs and vendors
    lrs = sorted(set(lr for _, lr in by_vendor_lr.keys()))
    vendors = sorted(set(v for v, _ in by_vendor_lr.keys()))
    
    # Separate vendors by dtype
    fp16_vendors = [v for v in vendors if 'fp16' in v]
    bf16_vendors = [v for v in vendors if 'bf16' in v]
    
    # Create subplots: 2 rows (FP16 and BF16), one column per LR
    n_lrs = len(lrs)
    fig, axes = plt.subplots(2, n_lrs, figsize=(6*n_lrs, 10))
    if n_lrs == 1:
        axes = axes.reshape(2, 1)
    
    # Color scheme for vendor types
    def get_vendor_color(vendor):
        """Assign colors based on vendor type."""
        if 'mi210' in vendor:
            return 'red'
        else:
            return 'green'  # base vendor
    
    def get_vendor_style(vendor):
        """Assign line styles based on FTZ/subnormals."""
        if 'mi210' in vendor:
            return '--'  # FTZ
        else:
            return '-'   # subnormals
    
    # Row 0: FP16 vendors
    for lr_idx, lr in enumerate(lrs):
        ax = axes[0, lr_idx]
        
        # Plot loss curves for each FP16 vendor at this LR
        for vendor in fp16_vendors:
            key = (vendor, lr)
            if key not in by_vendor_lr:
                continue
            
            runs = by_vendor_lr[key]
            if not runs:
                continue
            
            # Average losses across seeds
            max_steps = max(len(r["losses"]) for r in runs)
            avg_losses = np.zeros(max_steps)
            count = np.zeros(max_steps)
            
            for r in runs:
                losses = r["losses"]
                for i, loss in enumerate(losses):
                    if i < max_steps:
                        avg_losses[i] += loss
                        count[i] += 1
            
            # Compute average
            avg_losses = avg_losses / np.maximum(count, 1)
            steps = np.arange(1, len(avg_losses) + 1)
            
            # Plot curve with vendor-specific styling
            color = get_vendor_color(vendor)
            linestyle = get_vendor_style(vendor)
            # Simplify vendor name for legend
            vendor_label = vendor.replace('fp16_', '').replace('_a100', '').replace('_mi210', '')
            ax.plot(steps, avg_losses, label=vendor_label, linewidth=2, 
                   alpha=0.8, color=color, linestyle=linestyle)
            
            # Mark saturation point
            sat_step = detect_saturation(avg_losses.tolist())
            if sat_step is not None and sat_step < len(avg_losses):
                ax.axvline(x=sat_step, color=color, linestyle=':', alpha=0.5, linewidth=1)
                ax.plot(sat_step, avg_losses[sat_step], 'o', color=color, 
                       markersize=8, alpha=0.7, markeredgecolor='black', markeredgewidth=1)
        
        ax.set_xlabel("Step", fontsize=11)
        if lr_idx == 0:
            ax.set_ylabel("Loss (FP16)", fontsize=11, fontweight='bold')
        ax.set_title(f"LR={lr:.4f}", fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    # Row 1: BF16 vendors
    for lr_idx, lr in enumerate(lrs):
        ax = axes[1, lr_idx]
        
        # Plot loss curves for each BF16 vendor at this LR
        for vendor in bf16_vendors:
            key = (vendor, lr)
            if key not in by_vendor_lr:
                continue
            
            runs = by_vendor_lr[key]
            if not runs:
                continue
            
            # Average losses across seeds
            max_steps = max(len(r["losses"]) for r in runs)
            avg_losses = np.zeros(max_steps)
            count = np.zeros(max_steps)
            
            for r in runs:
                losses = r["losses"]
                for i, loss in enumerate(losses):
                    if i < max_steps:
                        avg_losses[i] += loss
                        count[i] += 1
            
            # Compute average
            avg_losses = avg_losses / np.maximum(count, 1)
            steps = np.arange(1, len(avg_losses) + 1)
            
            # Plot curve with vendor-specific styling
            color = get_vendor_color(vendor)
            linestyle = get_vendor_style(vendor)
            # Simplify vendor name for legend
            vendor_label = vendor.replace('bf16_', '').replace('_a100', '').replace('_mi210', '')
            ax.plot(steps, avg_losses, label=vendor_label, linewidth=2, 
                   alpha=0.8, color=color, linestyle=linestyle)
            
            # Mark saturation point
            sat_step = detect_saturation(avg_losses.tolist())
            if sat_step is not None and sat_step < len(avg_losses):
                ax.axvline(x=sat_step, color=color, linestyle=':', alpha=0.5, linewidth=1)
                ax.plot(sat_step, avg_losses[sat_step], 'o', color=color, 
                       markersize=8, alpha=0.7, markeredgecolor='black', markeredgewidth=1)
        
        ax.set_xlabel("Step", fontsize=11)
        if lr_idx == 0:
            ax.set_ylabel("Loss (BF16)", fontsize=11, fontweight='bold')
        ax.set_title(f"LR={lr:.4f}", fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    plt.suptitle("Loss Saturation by Dtype: FP16 (top) vs BF16 (bottom)\n" + 
                 "(Green=A100, Red=MI210)\n" + "(subscale=1e-4)", 
                 fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved saturation plot: {output_file}")


def plot_loss_saturation_comparison(results: List[Dict], output_file: str):
    """
    Plot a comparison of loss curves across all vendors at a specific LR (or best LR per vendor).
    Shows saturation points clearly.
    """
    # Group by vendor, find best LR (lowest final loss)
    by_vendor = defaultdict(list)
    for r in results:
        if "error" in r or r.get("diverged", False):
            continue
        if "losses" not in r or not r["losses"]:
            continue
        by_vendor[r["vendor"]].append(r)
    
    vendors = sorted(by_vendor.keys())
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    for vendor in vendors:
        runs = by_vendor[vendor]
        
        # Find best LR (lowest average final loss)
        by_lr = defaultdict(list)
        for r in runs:
            by_lr[r["lr"]].append(r)
        
        best_lr = None
        best_avg_final = float('inf')
        for lr, lr_runs in by_lr.items():
            final_losses = [r.get("final_loss", float('inf')) for r in lr_runs]
            avg_final = np.mean(final_losses) if final_losses else float('inf')
            if avg_final < best_avg_final:
                best_avg_final = avg_final
                best_lr = lr
        
        if best_lr is None:
            continue
        
        # Average losses across seeds for best LR
        best_runs = [r for r in runs if r["lr"] == best_lr]
        max_steps = max(len(r["losses"]) for r in best_runs)
        avg_losses = np.zeros(max_steps)
        count = np.zeros(max_steps)
        
        for r in best_runs:
            losses = r["losses"]
            for i, loss in enumerate(losses):
                if i < max_steps:
                    avg_losses[i] += loss
                    count[i] += 1
        
        avg_losses = avg_losses / np.maximum(count, 1)
        steps = np.arange(1, len(avg_losses) + 1)
        
        # Plot curve
        ax.plot(steps, avg_losses, label=f"{vendor} (LR={best_lr:.4f})", linewidth=2.5, alpha=0.8)
        
        # Mark saturation point
        sat_step = detect_saturation(avg_losses.tolist())
        if sat_step is not None and sat_step < len(avg_losses):
            ax.axvline(x=sat_step, color='gray', linestyle='--', alpha=0.4, linewidth=1)
            ax.plot(sat_step, avg_losses[sat_step], 'o', markersize=10, alpha=0.8, 
                   markeredgewidth=2, markeredgecolor='black')
    
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Loss Saturation Comparison Across Vendors (Best LR per Vendor)\n"
                + "(subscale=1e-4)", 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze LR sweep results")
    parser.add_argument("--result-file", type=str, required=True, help="LR sweep results pickle file")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Load results
    data = load_results(args.result_file)
    results = data["results"]
    config = data["config"]
    print("results = ")
    print(results)
    # Analyze
    analysis = analyze_stability(results)
    
    # Print summary
    print("=== LR Sweep Analysis ===\n")
    vendors = sorted(set(v for v, _ in analysis.keys()))
    lrs = sorted(set(lr for _, lr in analysis.keys()))
    
    for vendor in vendors:
        print(f"{vendor}:")
        for lr in lrs:
            key = (vendor, lr)
            if key in analysis:
                a = analysis[key]
                print(f"  LR {lr:.4f}: {a['diverged_count']}/{a['total_runs']} diverged "
                      f"({a['divergence_rate']:.1%}), "
                      f"avg_final={a['avg_final_loss']:.4f}")
        
        # Find critical LR (50% divergence threshold)
        critical = find_critical_lr(analysis, vendor, lrs, threshold=0.5)
        if critical:
            print(f"  → Critical LR (50% divergence): {critical:.4f}")
        print()
    
    # Generate plots
    output_dir = args.output_dir or "lr_sweep_analysis"
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    #lr sweep
    saturation_file = os.path.join(output_dir, "loss_saturation_analysis.png")
    plot_loss_saturation_analysis(results, saturation_file)
    
    # Generate saturation plots
    saturation_file = os.path.join(output_dir, "loss_saturation.png")
    plot_loss_saturation(results, saturation_file)
    
    comparison_file = os.path.join(output_dir, "loss_saturation_comparison.png")
    plot_loss_saturation_comparison(results, comparison_file)
    
    # Save analysis
    analysis_file = os.path.join(output_dir, "analysis.pkl")
    with open(analysis_file, "wb") as f:
        pickle.dump({"analysis": analysis, "config": config}, f)
    print(f"Saved analysis: {analysis_file}")


if __name__ == "__main__":
    main()



