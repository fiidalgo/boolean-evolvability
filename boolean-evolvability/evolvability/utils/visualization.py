import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Optional
import os


def plot_fitness_history(fitness_history: List[float], title: str = "Fitness over Generations",
                         save_path: Optional[str] = None):
    """
    Plot the fitness history over generations.
    
    Args:
        fitness_history: List of fitness values at each generation
        title: Plot title
        save_path: If provided, save the figure to this path
    """
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history, 'b-', linewidth=2)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add horizontal line at y=1.0 to show maximum fitness
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Maximum Fitness')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_experiment_results(results: Dict[str, Any], metric: str = 'success_rates',
                            title: Optional[str] = None, save_path: Optional[str] = None):
    """
    Plot experiment results across different n values.
    
    Args:
        results: Experiment results dictionary
        metric: Metric to plot ('success_rates', 'avg_generations', or 'avg_times')
        title: Plot title (if None, a default title will be used)
        save_path: If provided, save the figure to this path
    """
    valid_metrics = {'success_rates', 'avg_generations', 'avg_times'}
    if metric not in valid_metrics:
        raise ValueError(f"Invalid metric: {metric}. Must be one of {valid_metrics}")
    
    plt.figure(figsize=(10, 6))
    
    n_values = results['n_values']
    y_values = results[metric]
    
    plt.plot(n_values, y_values, 'o-', linewidth=2, markersize=8)
    
    plt.xlabel('Problem Size (n)')
    
    if metric == 'success_rates':
        plt.ylabel('Success Rate')
        plt.ylim(-0.05, 1.05)  # Ensure y-axis goes from 0 to 1
        if title is None:
            title = f"Success Rate for {results['function_class']} (ε={results['epsilon']})"
    elif metric == 'avg_generations':
        plt.ylabel('Average Generations')
        if title is None:
            title = f"Average Generations for {results['function_class']} (ε={results['epsilon']})"
    else:  # avg_times
        plt.ylabel('Average Time (seconds)')
        if title is None:
            title = f"Average Runtime for {results['function_class']} (ε={results['epsilon']})"
    
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_comparison(results_list: List[Dict[str, Any]], metric: str = 'success_rates',
                    title: Optional[str] = None, save_path: Optional[str] = None):
    """
    Plot comparison of multiple experiment results.
    
    Args:
        results_list: List of experiment results dictionaries
        metric: Metric to plot ('success_rates', 'avg_generations', or 'avg_times')
        title: Plot title (if None, a default title will be used)
        save_path: If provided, save the figure to this path
    """
    valid_metrics = {'success_rates', 'avg_generations', 'avg_times'}
    if metric not in valid_metrics:
        raise ValueError(f"Invalid metric: {metric}. Must be one of {valid_metrics}")
    
    plt.figure(figsize=(12, 7))
    
    # Define a color cycle for different function classes
    colors = plt.cm.tab10.colors
    
    for i, results in enumerate(results_list):
        color = colors[i % len(colors)]
        label = results['function_class']
        
        plt.plot(results['n_values'], results[metric], 'o-', 
                 linewidth=2, markersize=8, color=color, label=label)
    
    plt.xlabel('Problem Size (n)')
    
    if metric == 'success_rates':
        plt.ylabel('Success Rate')
        plt.ylim(-0.05, 1.05)  # Ensure y-axis goes from 0 to 1
        if title is None:
            title = f"Success Rate Comparison (ε={results_list[0]['epsilon']})"
    elif metric == 'avg_generations':
        plt.ylabel('Average Generations')
        if title is None:
            title = f"Average Generations Comparison (ε={results_list[0]['epsilon']})"
    else:  # avg_times
        plt.ylabel('Average Time (seconds)')
        if title is None:
            title = f"Average Runtime Comparison (ε={results_list[0]['epsilon']})"
    
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show() 