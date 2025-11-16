# merge_real_gpu_results.py
#
# Merge results from all real GPU job outputs into a single file for analysis.

import argparse
import glob
import os
import pickle
from typing import Dict, List


def merge_results(output_dir: str) -> Dict:
    """Load and merge all result files from real GPU jobs."""
    results_dir = os.path.join(output_dir, "results")
    if not os.path.isdir(results_dir):
        raise ValueError(f"Results directory does not exist: {results_dir}")
    
    pattern = os.path.join(results_dir, "result_*.pkl")
    result_files = glob.glob(pattern)
    
    if not result_files:
        raise ValueError(f"No result files found in {results_dir}. Pattern: {pattern}")
    
    all_results = []
    for result_file in result_files:
        try:
            with open(result_file, "rb") as f:
                result = pickle.load(f)
                all_results.append(result)
        except Exception as e:
            print(f"Warning: Failed to load {result_file}: {e}")
    
    print(f"Loaded {len(all_results)} results from {len(result_files)} files")
    
    # Load config if available
    config_file = os.path.join(output_dir, "config.txt")
    config = {}
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            for line in f:
                if ":" in line:
                    key, value = line.split(":", 1)
                    config[key.strip()] = value.strip()
    
    return {
        "results": all_results,
        "config": config,
    }


def main():
    parser = argparse.ArgumentParser(description="Merge real GPU job results")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory containing result files (default: latest timestamped directory)"
    )
    
    args = parser.parse_args()
    
    # Find output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Find latest timestamped directory
        dirs = glob.glob("real_gpu_results_*")
        if not dirs:
            raise ValueError("No output directories found. Specify --output-dir or run jobs first.")
        output_dir = max(dirs, key=os.path.getmtime)
        print(f"Using latest output directory: {output_dir}")
    
    if not os.path.isdir(output_dir):
        raise ValueError(f"Output directory does not exist: {output_dir}")
    
    print(f"\n=== Merging results from {output_dir} ===\n")
    
    # Merge results
    merged_data = merge_results(output_dir)
    
    # Save merged results
    merged_file = os.path.join(output_dir, "merged_results.pkl")
    with open(merged_file, "wb") as f:
        pickle.dump(merged_data, f)
    
    print(f"\n=== Merged results saved to {merged_file} ===")
    print(f"Total results: {len(merged_data['results'])}")
    
    # Print summary
    print("\n=== Summary by GPU Type and Precision ===")
    summary = {}
    for r in merged_data['results']:
        if 'error' in r:
            continue
        # Extract GPU type from job name or vendor
        vendor = r.get('vendor', 'unknown')
        precision = r.get('precision', 'unknown')
        key = f"{vendor}_{precision}"
        if key not in summary:
            summary[key] = {'total': 0, 'diverged': 0, 'converged': 0}
        summary[key]['total'] += 1
        if r.get('diverged', False):
            summary[key]['diverged'] += 1
        else:
            summary[key]['converged'] += 1
    
    for key, stats in sorted(summary.items()):
        print(f"{key}: {stats['converged']}/{stats['total']} converged, {stats['diverged']} diverged")
    
    print(f"\nResults ready for analysis in: {merged_file}")


if __name__ == "__main__":
    main()

