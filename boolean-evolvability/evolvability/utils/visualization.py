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
    valid_metrics = {'success_rates', 'avg_generations', 'avg_times', 
                     'avg_beneficial_mutations', 'avg_neutral_mutations'}
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
    elif metric == 'avg_times':
        plt.ylabel('Average Time (seconds)')
        if title is None:
            title = f"Average Runtime for {results['function_class']} (ε={results['epsilon']})"
    elif metric == 'avg_beneficial_mutations':
        plt.ylabel('Average Beneficial Mutations')
        if title is None:
            title = f"Average Beneficial Mutations for {results['function_class']} (ε={results['epsilon']})"
    elif metric == 'avg_neutral_mutations':
        plt.ylabel('Average Neutral Mutations')
        if title is None:
            title = f"Average Neutral Mutations for {results['function_class']} (ε={results['epsilon']})"
    
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
    valid_metrics = {'success_rates', 'avg_generations', 'avg_times',
                     'avg_beneficial_mutations', 'avg_neutral_mutations'}
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
    elif metric == 'avg_times':
        plt.ylabel('Average Time (seconds)')
        if title is None:
            title = f"Average Runtime Comparison (ε={results_list[0]['epsilon']})"
    elif metric == 'avg_beneficial_mutations':
        plt.ylabel('Average Beneficial Mutations')
        if title is None:
            title = f"Average Beneficial Mutations Comparison (ε={results_list[0]['epsilon']})"
    elif metric == 'avg_neutral_mutations':
        plt.ylabel('Average Neutral Mutations')
        if title is None:
            title = f"Average Neutral Mutations Comparison (ε={results_list[0]['epsilon']})"
    
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


def plot_fitness_over_generations(results: Dict[str, Any], 
                                  title: Optional[str] = None,
                                  save_path: Optional[str] = None):
    """
    Plot average fitness over generations for each problem size.
    
    Args:
        results: Experiment results dictionary containing fitness_histories
        title: Plot title (if None, a default title will be used)
        save_path: If provided, save the figure to this path
    """
    if 'fitness_histories' not in results:
        raise ValueError("Results dictionary must contain 'fitness_histories' key")
    
    plt.figure(figsize=(12, 7))
    
    # Define a color cycle for different n values
    colors = plt.cm.tab10.colors
    
    # Get maximum length across all histories
    max_len = 0
    for n, histories in results['fitness_histories'].items():
        for history in histories:
            max_len = max(max_len, len(history))
    
    # Plot average fitness history for each n
    for i, (n, histories) in enumerate(sorted(results['fitness_histories'].items())):
        if not histories:
            continue
            
        # For each generation, compute average fitness across all trials
        avg_history = np.zeros(max_len)
        count = np.zeros(max_len)
        
        for history in histories:
            # Pad history with final value if needed
            padded = history + [history[-1]] * (max_len - len(history))
            for j, fitness in enumerate(padded):
                avg_history[j] += fitness
                count[j] += 1
        
        # Compute average (handling potential division by zero)
        avg_history = np.divide(avg_history, count, out=np.zeros_like(avg_history), where=count!=0)
        
        # Plot this n's average history
        color = colors[i % len(colors)]
        plt.plot(range(max_len), avg_history, '-', 
                 linewidth=2, color=color, label=f'n={n}')
    
    plt.xlabel('Generation')
    plt.ylabel('Average Fitness')
    
    # Add horizontal line at y=1-epsilon to show success threshold
    plt.axhline(y=1-results['epsilon'], color='r', linestyle='--', 
                alpha=0.5, label=f'Success Threshold (1-ε)')
    
    if title is None:
        title = f"Average Fitness over Generations for {results['function_class']}"
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


def plot_mutation_counts(results: Dict[str, Any], 
                         title: Optional[str] = None,
                         save_path: Optional[str] = None):
    """
    Plot beneficial vs. neutral mutation counts for each problem size.
    
    Args:
        results: Experiment results dictionary
        title: Plot title (if None, a default title will be used)
        save_path: If provided, save the figure to this path
    """
    if 'avg_beneficial_mutations' not in results or 'avg_neutral_mutations' not in results:
        raise ValueError("Results dictionary must contain mutation count data")
    
    plt.figure(figsize=(10, 6))
    
    n_values = results['n_values']
    beneficial = results['avg_beneficial_mutations']
    neutral = results['avg_neutral_mutations']
    
    width = 0.35  # the width of the bars
    x = np.arange(len(n_values))
    
    plt.bar(x - width/2, beneficial, width, label='Beneficial', color='green', alpha=0.7)
    plt.bar(x + width/2, neutral, width, label='Neutral', color='blue', alpha=0.7)
    
    plt.xlabel('Problem Size (n)')
    plt.ylabel('Average Count per Run')
    plt.xticks(x, [str(n) for n in n_values])
    
    if title is None:
        title = f"Mutation Counts for {results['function_class']} (ε={results['epsilon']})"
    plt.title(title)
    
    plt.grid(True, linestyle='--', alpha=0.4, axis='y')
    plt.legend(loc='best')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_mutation_comparison(results_list: List[Dict[str, Any]],
                             n_value: int,
                             title: Optional[str] = None,
                             save_path: Optional[str] = None):
    """
    Compare beneficial vs. neutral mutations across function classes for a specific n value.
    
    Args:
        results_list: List of experiment results dictionaries
        n_value: Problem size to compare
        title: Plot title (if None, a default title will be used)
        save_path: If provided, save the figure to this path
    """
    plt.figure(figsize=(12, 7))
    
    function_classes = [results['function_class'] for results in results_list]
    beneficial = []
    neutral = []
    
    for results in results_list:
        if n_value not in results['n_values']:
            raise ValueError(f"Problem size n={n_value} not found in results for {results['function_class']}")
        
        idx = results['n_values'].index(n_value)
        beneficial.append(results['avg_beneficial_mutations'][idx])
        neutral.append(results['avg_neutral_mutations'][idx])
    
    width = 0.35  # the width of the bars
    x = np.arange(len(function_classes))
    
    plt.bar(x - width/2, beneficial, width, label='Beneficial', color='green', alpha=0.7)
    plt.bar(x + width/2, neutral, width, label='Neutral', color='blue', alpha=0.7)
    
    plt.xlabel('Function Class')
    plt.ylabel('Average Count per Run')
    plt.xticks(x, function_classes, rotation=45)
    
    if title is None:
        title = f"Mutation Counts Comparison for n={n_value} (ε={results_list[0]['epsilon']})"
    plt.title(title)
    
    plt.grid(True, linestyle='--', alpha=0.4, axis='y')
    plt.legend(loc='best')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_experiment_type_comparison(standard_results: List[Dict[str, Any]],
                              distribution_results: List[Dict[str, Any]],
                              no_neutral_results: List[Dict[str, Any]],
                              smart_init_results: List[Dict[str, Any]],
                              function_class: str,
                              metric: str = 'success_rates',
                              title: Optional[str] = None, 
                              save_path: Optional[str] = None):
    """
    Plot comparison of the same function class across different experiment types.
    
    Args:
        standard_results: List of standard experiment results
        distribution_results: List of distribution experiment results
        no_neutral_results: List of no-neutral-mutation experiment results
        smart_init_results: List of smart-initialization experiment results
        function_class: Name of the function class to compare
        metric: Metric to plot
        title: Plot title
        save_path: If provided, save the figure to this path
    """
    valid_metrics = {'success_rates', 'avg_generations', 'avg_times',
                    'avg_beneficial_mutations', 'avg_neutral_mutations'}
    if metric not in valid_metrics:
        raise ValueError(f"Invalid metric: {metric}. Must be one of {valid_metrics}")
    
    plt.figure(figsize=(12, 7))
    
    # Define experiment types and their data
    experiment_types = [
        ("Standard", standard_results, 'b'),
        ("No Neutral", no_neutral_results, 'r'),
        ("Smart Init", smart_init_results, 'g'),
    ]
    
    # Add distribution experiments if available
    distribution_labels = {
        'biased_075': "Biased 0.75",
        'binomial': "Binomial",
        'beta': "Beta",
    }
    
    # Track if we found a matching function class
    found_class = False
    
    # Plot standard, no-neutral, and smart-init experiments
    for label, results_list, color in experiment_types:
        for results in results_list:
            if results['function_class'] == function_class:
                found_class = True
                plt.plot(results['n_values'], results[metric], 'o-', 
                         linewidth=2, markersize=8, color=color, label=label)
                break
    
    # Plot distribution experiments with different markers
    markers = ['s--', '^--', 'v--']  # Square, triangle up, triangle down
    colors = ['orange', 'purple', 'brown']
    
    for i, (dist_name, marker, color) in enumerate(zip(distribution_labels.keys(), markers, colors)):
        for results in distribution_results:
            if results['function_class'] == function_class and results.get('dist_name') == dist_name:
                found_class = True
                plt.plot(results['n_values'], results[metric], marker, 
                         linewidth=2, markersize=8, color=color, label=distribution_labels[dist_name])
                break
    
    if not found_class:
        plt.text(0.5, 0.5, f"No data for {function_class}", 
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=14)
    
    plt.xlabel('Problem Size (n)')
    
    if metric == 'success_rates':
        plt.ylabel('Success Rate')
        plt.ylim(-0.05, 1.05)
        if title is None:
            title = f"Success Rate Comparison for {function_class}"
    elif metric == 'avg_generations':
        plt.ylabel('Average Generations')
        if title is None:
            title = f"Average Generations Comparison for {function_class}"
    elif metric == 'avg_times':
        plt.ylabel('Average Time (seconds)')
        if title is None:
            title = f"Average Runtime Comparison for {function_class}"
    elif metric == 'avg_beneficial_mutations':
        plt.ylabel('Average Beneficial Mutations')
        if title is None:
            title = f"Average Beneficial Mutations Comparison for {function_class}"
    elif metric == 'avg_neutral_mutations':
        plt.ylabel('Average Neutral Mutations')
        if title is None:
            title = f"Average Neutral Mutations Comparison for {function_class}"
    
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