#!/usr/bin/env python3
"""
Main script for running evolvability experiments.

This implements Phase 1 of the Boolean evolvability project, testing whether
different types of Boolean functions can evolve under Valiant's model.
"""

import argparse
import os
import time
import numpy as np
from typing import List, Dict, Any

from evolvability.functions import (
    MonotoneConjunction, 
    MonotoneDisjunction,
    GeneralConjunction,
    GeneralDisjunction,
    Parity, 
    Majority
)
from evolvability.environment import Environment
from evolvability.evolve import EvolutionaryAlgorithm, run_experiment
from evolvability.utils.visualization import (
    plot_fitness_history, 
    plot_experiment_results, 
    plot_comparison,
    plot_fitness_over_generations,
    plot_mutation_counts,
    plot_mutation_comparison
)
from evolvability.utils.io import (
    save_results_json, 
    create_results_dir
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run evolvability experiments')
    
    parser.add_argument(
        '--function-classes', 
        nargs='+',
        choices=[
            'conjunction', 
            'disjunction', 
            'general_conjunction',
            'general_disjunction',
            'parity', 
            'majority'
        ],
        default=['conjunction', 'disjunction', 'parity'],
        help='List of function classes to test'
    )
    
    parser.add_argument(
        '--n-values', 
        nargs='+',
        type=int,
        default=[10, 20, 50],
        help='List of input sizes to test'
    )
    
    parser.add_argument(
        '--trials', 
        type=int,
        default=5,
        help='Number of trials for each configuration'
    )
    
    parser.add_argument(
        '--epsilon', 
        type=float,
        default=0.05,
        help='Target error threshold'
    )
    
    parser.add_argument(
        '--sample-size', 
        type=int,
        default=1000,
        help='Number of examples to use for fitness evaluation'
    )
    
    parser.add_argument(
        '--validation-size', 
        type=int,
        default=5000,
        help='Number of examples to use for final validation'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Print verbose output'
    )
    
    parser.add_argument(
        '--single-run', 
        action='store_true',
        help='Run a single experiment (for quick testing)'
    )
    
    return parser.parse_args()


def get_function_class(name: str):
    """Get the class associated with a function name."""
    function_classes = {
        'conjunction': MonotoneConjunction,
        'disjunction': MonotoneDisjunction,
        'general_conjunction': GeneralConjunction,
        'general_disjunction': GeneralDisjunction,
        'parity': Parity,
        'majority': Majority
    }
    return function_classes[name]


def run_single_experiment(args):
    """Run a single experiment for demonstration/testing."""
    print("Running single experiment for testing...")
    
    # Choose a function class
    function_class = MonotoneConjunction
    n = 10
    
    # Create a target function with a specific set of variables
    target_vars = {0, 2, 5}  # Use variables x1, x3, and x6
    target = function_class(n, target_vars)
    print(f"Target function: {target}")
    
    # Create an initial hypothesis (randomly)
    initial_hypothesis = function_class(n)
    print(f"Initial hypothesis: {initial_hypothesis}")
    
    # Create environment and algorithm
    env = Environment(n, target)
    algo = EvolutionaryAlgorithm(
        environment=env,
        initial_hypothesis=initial_hypothesis,
        epsilon=args.epsilon,
        sample_size=args.sample_size,
        validation_size=args.validation_size,
        max_generations=1000,
        stagnation_threshold=50
    )
    
    # Run the algorithm
    print("Running evolutionary algorithm...")
    result = algo.run(verbose=True)
    
    # Print results
    print("\nResults:")
    print(f"Success: {result.success}")
    print(f"Generations: {result.generations}")
    print(f"Final fitness: {result.final_fitness:.4f}")
    print(f"Final hypothesis: {result.final_hypothesis}")
    print(f"Beneficial mutations: {result.beneficial_mutations}")
    print(f"Neutral mutations: {result.neutral_mutations}")
    
    # Plot fitness history
    results_dir = create_results_dir("single_experiment")
    plot_fitness_history(
        result.fitness_history,
        title=f"Fitness History for {function_class.__name__}",
        save_path=f"{results_dir}/fitness_history.png"
    )
    
    return result


def run_full_experiments(args):
    """Run full experiments on all specified function classes and sizes."""
    print("Running full experiments...")
    
    results_dir = create_results_dir("full_experiments")
    all_results = []
    
    # Run experiments for each function class
    for function_name in args.function_classes:
        function_class = get_function_class(function_name)
        print(f"\n=== Testing {function_class.__name__} ===")
        
        results = run_experiment(
            function_class=function_class,
            n_values=args.n_values,
            num_trials=args.trials,
            epsilon=args.epsilon,
            sample_size=args.sample_size,
            validation_size=args.validation_size,
            verbose=args.verbose
        )
        
        all_results.append(results)
        
        # Save individual results
        save_results_json(
            results,
            f"{results_dir}/{function_class.__name__}_results.json"
        )
        
        # Plot individual results - standard metrics
        plot_experiment_results(
            results,
            metric='success_rates',
            save_path=f"{results_dir}/{function_class.__name__}_success_rate.png"
        )
        
        plot_experiment_results(
            results,
            metric='avg_generations',
            save_path=f"{results_dir}/{function_class.__name__}_generations.png"
        )
        
        # Plot new metrics - fitness over generations and mutation counts
        plot_fitness_over_generations(
            results,
            title=f"Fitness over Generations for {function_class.__name__}",
            save_path=f"{results_dir}/{function_class.__name__}_fitness_over_generations.png"
        )
        
        plot_mutation_counts(
            results,
            title=f"Mutation Counts for {function_class.__name__}",
            save_path=f"{results_dir}/{function_class.__name__}_mutation_counts.png"
        )
    
    # Plot comparisons - standard metrics
    plot_comparison(
        all_results,
        metric='success_rates',
        title="Success Rate Comparison",
        save_path=f"{results_dir}/comparison_success_rate.png"
    )
    
    plot_comparison(
        all_results,
        metric='avg_generations',
        title="Average Generations Comparison",
        save_path=f"{results_dir}/comparison_generations.png"
    )
    
    # Plot comparisons - new metrics
    plot_comparison(
        all_results,
        metric='avg_beneficial_mutations',
        title="Beneficial Mutations Comparison",
        save_path=f"{results_dir}/comparison_beneficial_mutations.png"
    )
    
    plot_comparison(
        all_results,
        metric='avg_neutral_mutations',
        title="Neutral Mutations Comparison",
        save_path=f"{results_dir}/comparison_neutral_mutations.png"
    )
    
    # Plot mutation comparison for a specific problem size (middle one)
    middle_n = args.n_values[len(args.n_values) // 2]
    plot_mutation_comparison(
        all_results,
        n_value=middle_n,
        title=f"Mutation Types Comparison (n={middle_n})",
        save_path=f"{results_dir}/mutation_types_n{middle_n}.png"
    )
    
    return all_results


def main():
    """Main function to run experiments."""
    args = parse_args()
    
    start_time = time.time()
    
    if args.single_run:
        result = run_single_experiment(args)
    else:
        results = run_full_experiments(args)
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main() 