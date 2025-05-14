#!/usr/bin/env python3
"""
Script to generate a comprehensive report from experiment results.

This aggregates results from all experiments and creates summary
plots comparing different experiment configurations.
"""

import os
import sys
import json
import argparse
import glob
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt

from evolvability.functions import (
    MonotoneConjunction, 
    MonotoneDisjunction,
    GeneralConjunction,
    GeneralDisjunction,
    Parity, 
    Majority
)
from evolvability.utils.visualization import (
    plot_comparison,
    plot_experiment_type_comparison
)

# Function class name mapping
FUNCTION_CLASSES = {
    'MonotoneConjunction': 'Monotone Conjunction',
    'MonotoneDisjunction': 'Monotone Disjunction',
    'GeneralConjunction': 'General Conjunction',
    'GeneralDisjunction': 'General Disjunction',
    'Parity': 'Parity',
    'Majority': 'Majority'
}

def load_results(results_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load all results from the specified directory.
    
    Args:
        results_dir: Path to the directory containing results
        
    Returns:
        Dictionary mapping experiment type to list of results
    """
    if not os.path.exists(results_dir):
        print(f"Error: Results directory '{results_dir}' not found.")
        sys.exit(1)
    
    # Categorize results by experiment type
    results = {
        'standard': [],
        'binomial': [],
        'beta': [],
        'biased_075': [],
        'no_neutral': [],
        'smart_init': []
    }
    
    # Find all JSON result files
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    
    for file_path in json_files:
        file_name = os.path.basename(file_path)
        
        # Load the results
        with open(file_path, 'r') as f:
            result = json.load(f)
        
        # Categorize by filename prefix
        if file_name.startswith('standard_'):
            results['standard'].append(result)
        elif file_name.startswith('binomial_'):
            results['binomial'].append(result)
        elif file_name.startswith('beta_'):
            results['beta'].append(result)
        elif file_name.startswith('biased_075_'):
            results['biased_075'].append(result)
        elif file_name.startswith('no_neutral_'):
            results['no_neutral'].append(result)
        elif file_name.startswith('smart_init_'):
            results['smart_init'].append(result)
    
    return results


def generate_summary_plots(results: Dict[str, List[Dict[str, Any]]], output_dir: str):
    """
    Generate summary plots comparing results across different experiment types.
    
    Args:
        results: Dictionary mapping experiment type to list of results
        output_dir: Directory to save the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all function classes from standard results
    function_classes = [r['function_class'] for r in results['standard']]
    
    # Generate plots for each metric
    metrics = [
        ('success_rates', 'Success Rate'),
        ('avg_generations', 'Average Generations'),
        ('avg_beneficial_mutations', 'Average Beneficial Mutations'),
        ('avg_neutral_mutations', 'Average Neutral Mutations')
    ]
    
    # 1. Generate comparison plots for each function class across experiment types
    for function_class in function_classes:
        for metric, metric_name in metrics:
            plot_experiment_type_comparison(
                standard_results=results['standard'],
                distribution_results=results['binomial'] + results['beta'] + results['biased_075'],
                no_neutral_results=results['no_neutral'],
                smart_init_results=results['smart_init'],
                function_class=function_class,
                metric=metric,
                title=f"{metric_name} Comparison for {FUNCTION_CLASSES.get(function_class, function_class)}",
                save_path=f"{output_dir}/comparison_{function_class}_{metric}.png"
            )
    
    # 2. Generate function class comparison within each experiment type
    experiment_types = [
        ('standard', 'Standard'),
        ('no_neutral', 'No Neutral Mutations'),
        ('smart_init', 'Smart Initialization')
    ]
    
    for exp_type, exp_name in experiment_types:
        if not results[exp_type]:
            continue
            
        for metric, metric_name in metrics:
            plot_comparison(
                results[exp_type],
                metric=metric,
                title=f"{metric_name} Comparison ({exp_name})",
                save_path=f"{output_dir}/{exp_type}_comparison_{metric}.png"
            )
    
    # 3. Generate distribution comparison plots
    for metric, metric_name in metrics:
        # Skip if no distribution results
        if not (results['binomial'] or results['beta'] or results['biased_075']):
            continue
            
        distribution_results = []
        for dist_type, dist_label in [
            ('binomial', 'Binomial'),
            ('beta', 'Beta'),
            ('biased_075', 'Biased 0.75')
        ]:
            # Add distribution type to the function class name for clearer legend
            for r in results[dist_type]:
                r_copy = r.copy()
                r_copy['function_class'] = f"{r['function_class']} ({dist_label})"
                distribution_results.append(r_copy)
        
        plot_comparison(
            distribution_results,
            metric=metric,
            title=f"{metric_name} Comparison (Distributions)",
            save_path=f"{output_dir}/distribution_comparison_{metric}.png"
        )


def generate_report(results_dir: str, output_dir: str):
    """
    Generate a comprehensive report from experiment results.
    
    Args:
        results_dir: Directory containing experiment results
        output_dir: Directory to save the report
    """
    # Load results
    results = load_results(results_dir)
    
    # Generate summary plots
    generate_summary_plots(results, output_dir)
    
    # Write summary text report
    with open(os.path.join(output_dir, "report_summary.txt"), 'w') as f:
        f.write("Evolvability Experiments Summary Report\n")
        f.write("=====================================\n\n")
        
        # Count experiments
        total_experiments = sum(len(exp_results) for exp_results in results.values())
        f.write(f"Total experiments analyzed: {total_experiments}\n\n")
        
        # Standard experiments summary
        if results['standard']:
            f.write("Standard Experiments\n")
            f.write("-----------------\n")
            for result in results['standard']:
                success_rate = result['success_rates'][-1] if result['success_rates'] else 'N/A'
                avg_gens = result['avg_generations'][-1] if result['avg_generations'] else 'N/A'
                f.write(f"{result['function_class']}: ")
                f.write(f"Success rate: {success_rate:.2f}, ")
                f.write(f"Avg generations: {avg_gens:.1f}\n")
            f.write("\n")
        
        # No neutral mutations experiments summary
        if results['no_neutral']:
            f.write("No Neutral Mutations Experiments\n")
            f.write("-------------------------------\n")
            for result in results['no_neutral']:
                success_rate = result['success_rates'][-1] if result['success_rates'] else 'N/A'
                avg_gens = result['avg_generations'][-1] if result['avg_generations'] else 'N/A'
                f.write(f"{result['function_class']}: ")
                f.write(f"Success rate: {success_rate:.2f}, ")
                f.write(f"Avg generations: {avg_gens:.1f}\n")
            f.write("\n")
        
        # Smart initialization experiments summary
        if results['smart_init']:
            f.write("Smart Initialization Experiments\n")
            f.write("------------------------------\n")
            for result in results['smart_init']:
                success_rate = result['success_rates'][-1] if result['success_rates'] else 'N/A'
                avg_gens = result['avg_generations'][-1] if result['avg_generations'] else 'N/A'
                f.write(f"{result['function_class']}: ")
                f.write(f"Success rate: {success_rate:.2f}, ")
                f.write(f"Avg generations: {avg_gens:.1f}\n")
            f.write("\n")
        
        # Distribution experiments summary
        for dist_type, dist_name in [
            ('binomial', 'Binomial'),
            ('beta', 'Beta'),
            ('biased_075', 'Biased 0.75')
        ]:
            if results[dist_type]:
                f.write(f"{dist_name} Distribution Experiments\n")
                f.write(f"{'-' * (len(dist_name) + 24)}\n")
                for result in results[dist_type]:
                    success_rate = result['success_rates'][-1] if result['success_rates'] else 'N/A'
                    avg_gens = result['avg_generations'][-1] if result['avg_generations'] else 'N/A'
                    f.write(f"{result['function_class']}: ")
                    f.write(f"Success rate: {success_rate:.2f}, ")
                    f.write(f"Avg generations: {avg_gens:.1f}\n")
                f.write("\n")
    
    print(f"Report generated in {output_dir}")
    print(f"Summary report: {os.path.join(output_dir, 'report_summary.txt')}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate report from experiment results')
    
    parser.add_argument(
        '--results-dir',
        type=str,
        required=True,
        help='Directory containing experiment results'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='report',
        help='Directory to save the report (default: report)'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_report(args.results_dir, args.output_dir) 