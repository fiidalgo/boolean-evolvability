#!/usr/bin/env python3
"""
Script to run a comprehensive suite of evolvability experiments.

This script runs multiple experiment configurations:
1. Regular Tests: All function classes with standard parameters
2. Distribution Tests: Non-parity functions with different distributions
3. No-Neutral Tests: All function classes without neutral mutations
4. Smart-Init Tests: All function classes with intelligently pre-selected initial hypotheses
"""

import os
import time
import argparse
import json
import numpy as np
from typing import List, Dict, Any, Set

from evolvability.functions import (
    BooleanFunction,
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

# Constants for the experiments
DEFAULT_N_VALUES = [5, 10, 20, 30, 50]
DEFAULT_TRIALS = 5
DEFAULT_EPSILON = 0.05
DEFAULT_TOLERANCE = 0.01
DEFAULT_SAMPLE_SIZE = 1000
DEFAULT_VALIDATION_SIZE = 5000

# Function class mappings
FUNCTION_CLASSES = {
    'conjunction': MonotoneConjunction,
    'disjunction': MonotoneDisjunction,
    'general_conjunction': GeneralConjunction,
    'general_disjunction': GeneralDisjunction,
    'parity': Parity,
    'majority': Majority
}

def run_standard_tests(results_dir: str, n_values: List[int], trials: int, verbose: bool):
    """Run standard tests on all function classes."""
    print("\n===== Running Standard Tests =====")
    
    function_names = list(FUNCTION_CLASSES.keys())
    all_results = []
    
    for function_name in function_names:
        function_class = FUNCTION_CLASSES[function_name]
        print(f"\n--- Testing {function_class.__name__} ---")
        
        results = run_experiment(
            function_class=function_class,
            n_values=n_values,
            num_trials=trials,
            epsilon=DEFAULT_EPSILON,
            tolerance=DEFAULT_TOLERANCE,
            sample_size=DEFAULT_SAMPLE_SIZE,
            validation_size=DEFAULT_VALIDATION_SIZE,
            verbose=verbose
        )
        
        all_results.append(results)
        
        # Save individual results
        save_results_json(
            results,
            f"{results_dir}/standard_{function_class.__name__}_results.json"
        )
        
        # Plot individual results
        plot_experiment_results(
            results,
            metric='success_rates',
            save_path=f"{results_dir}/standard_{function_class.__name__}_success_rate.png"
        )
        
        plot_experiment_results(
            results,
            metric='avg_generations',
            save_path=f"{results_dir}/standard_{function_class.__name__}_generations.png"
        )
        
        plot_fitness_over_generations(
            results,
            title=f"Fitness over Generations for {function_class.__name__}",
            save_path=f"{results_dir}/standard_{function_class.__name__}_fitness_over_generations.png"
        )
        
        plot_mutation_counts(
            results,
            title=f"Mutation Counts for {function_class.__name__}",
            save_path=f"{results_dir}/standard_{function_class.__name__}_mutation_counts.png"
        )
    
    # Plot comparisons
    plot_comparison(
        all_results,
        metric='success_rates',
        title="Success Rate Comparison (Standard)",
        save_path=f"{results_dir}/standard_comparison_success_rate.png"
    )
    
    plot_comparison(
        all_results,
        metric='avg_generations',
        title="Average Generations Comparison (Standard)",
        save_path=f"{results_dir}/standard_comparison_generations.png"
    )

    plot_comparison(
        all_results,
        metric='avg_beneficial_mutations',
        title="Beneficial Mutations Comparison (Standard)",
        save_path=f"{results_dir}/standard_comparison_beneficial_mutations.png"
    )
    
    # Middle n value for mutation type comparison
    middle_n = n_values[len(n_values) // 2]
    plot_mutation_comparison(
        all_results,
        n_value=middle_n,
        title=f"Mutation Types Comparison (n={middle_n}, Standard)",
        save_path=f"{results_dir}/standard_mutation_types_n{middle_n}.png"
    )
    
    return all_results


def run_distribution_tests(results_dir: str, n_values: List[int], trials: int, verbose: bool):
    """Run tests with different distributions."""
    print("\n===== Running Distribution Tests =====")
    
    # Exclude parity for distribution tests
    function_names = [name for name in FUNCTION_CLASSES.keys() if name != 'parity']
    
    # Define different distributions to test
    distributions = [
        ('binomial', 'binomial', None),  # Binomial distribution with p=0.3, n=3
        ('beta', 'beta', None),          # Beta distribution with alpha=2, beta=5
        ('biased_075', 'biased', 0.75),  # Biased Bernoulli with p=0.75
    ]
    
    for dist_name, dist_type, bias in distributions:
        print(f"\n----- Testing with {dist_name} distribution -----")
        all_results = []
        
        for function_name in function_names:
            function_class = FUNCTION_CLASSES[function_name]
            print(f"\n--- Testing {function_class.__name__} ---")
            
            # Custom run_experiment for distributions
            results = run_distribution_experiment(
                function_class=function_class,
                n_values=n_values,
                num_trials=trials,
                distribution=dist_type,
                dist_name=dist_name,
                bias=bias,
                epsilon=DEFAULT_EPSILON,
                tolerance=DEFAULT_TOLERANCE,
                sample_size=DEFAULT_SAMPLE_SIZE,
                validation_size=DEFAULT_VALIDATION_SIZE,
                verbose=verbose
            )
            
            all_results.append(results)
            
            # Save individual results
            save_results_json(
                results,
                f"{results_dir}/{dist_name}_{function_class.__name__}_results.json"
            )
            
            # Plot individual results
            plot_experiment_results(
                results,
                metric='success_rates',
                save_path=f"{results_dir}/{dist_name}_{function_class.__name__}_success_rate.png"
            )
        
        # Plot comparisons for this distribution
        plot_comparison(
            all_results,
            metric='success_rates',
            title=f"Success Rate Comparison ({dist_name})",
            save_path=f"{results_dir}/{dist_name}_comparison_success_rate.png"
        )
        
        plot_comparison(
            all_results,
            metric='avg_generations',
            title=f"Average Generations Comparison ({dist_name})",
            save_path=f"{results_dir}/{dist_name}_comparison_generations.png"
        )


def run_no_neutral_tests(results_dir: str, n_values: List[int], trials: int, verbose: bool):
    """Run tests without allowing neutral mutations."""
    print("\n===== Running No-Neutral-Mutation Tests =====")
    
    function_names = list(FUNCTION_CLASSES.keys())
    all_results = []
    
    for function_name in function_names:
        function_class = FUNCTION_CLASSES[function_name]
        print(f"\n--- Testing {function_class.__name__} ---")
        
        results = run_no_neutral_experiment(
            function_class=function_class,
            n_values=n_values,
            num_trials=trials,
            epsilon=DEFAULT_EPSILON,
            tolerance=DEFAULT_TOLERANCE,
            sample_size=DEFAULT_SAMPLE_SIZE,
            validation_size=DEFAULT_VALIDATION_SIZE,
            verbose=verbose
        )
        
        all_results.append(results)
        
        # Save individual results
        save_results_json(
            results,
            f"{results_dir}/no_neutral_{function_class.__name__}_results.json"
        )
        
        # Plot individual results
        plot_experiment_results(
            results,
            metric='success_rates',
            save_path=f"{results_dir}/no_neutral_{function_class.__name__}_success_rate.png"
        )
    
    # Plot comparisons
    plot_comparison(
        all_results,
        metric='success_rates',
        title="Success Rate Comparison (No Neutral Mutations)",
        save_path=f"{results_dir}/no_neutral_comparison_success_rate.png"
    )
    
    plot_comparison(
        all_results,
        metric='avg_generations',
        title="Average Generations Comparison (No Neutral Mutations)",
        save_path=f"{results_dir}/no_neutral_comparison_generations.png"
    )


def run_smart_init_tests(results_dir: str, n_values: List[int], trials: int, verbose: bool):
    """Run tests with smartly pre-selected initial hypotheses."""
    print("\n===== Running Smart Initialization Tests =====")
    
    function_names = list(FUNCTION_CLASSES.keys())
    all_results = []
    
    for function_name in function_names:
        function_class = FUNCTION_CLASSES[function_name]
        print(f"\n--- Testing {function_class.__name__} ---")
        
        results = run_smart_init_experiment(
            function_class=function_class,
            n_values=n_values,
            num_trials=trials,
            epsilon=DEFAULT_EPSILON,
            tolerance=DEFAULT_TOLERANCE,
            sample_size=DEFAULT_SAMPLE_SIZE,
            validation_size=DEFAULT_VALIDATION_SIZE,
            verbose=verbose
        )
        
        all_results.append(results)
        
        # Save individual results
        save_results_json(
            results,
            f"{results_dir}/smart_init_{function_class.__name__}_results.json"
        )
        
        # Plot individual results
        plot_experiment_results(
            results,
            metric='success_rates',
            save_path=f"{results_dir}/smart_init_{function_class.__name__}_success_rate.png"
        )
    
    # Plot comparisons
    plot_comparison(
        all_results,
        metric='success_rates',
        title="Success Rate Comparison (Smart Initialization)",
        save_path=f"{results_dir}/smart_init_comparison_success_rate.png"
    )
    
    plot_comparison(
        all_results,
        metric='avg_generations',
        title="Average Generations Comparison (Smart Initialization)",
        save_path=f"{results_dir}/smart_init_comparison_generations.png"
    )


def run_distribution_experiment(function_class, n_values: List[int], num_trials: int = 10, 
                                distribution: str = 'biased', dist_name: str = 'biased', bias = 0.25,
                                epsilon: float = 0.05, tolerance: float = 0.01, sample_size: int = 1000,
                                validation_size: int = 5000, verbose: bool = False) -> Dict[str, Any]:
    """
    Run an experiment with a specific distribution.
    """
    results = {
        'function_class': function_class.__name__,
        'distribution': distribution,
        'dist_name': dist_name,
        'bias': bias,
        'epsilon': epsilon,
        'tolerance': tolerance,
        'trials': num_trials,
        'n_values': n_values,
        'success_rates': [],
        'avg_generations': [],
        'avg_times': [],
        'avg_beneficial_mutations': [],
        'avg_neutral_mutations': []
    }
    
    # For storing fitness history data for plotting
    fitness_histories = {n: [] for n in n_values}
    
    for n in n_values:
        if verbose:
            print(f"\n--- Running experiments for n={n} ---")
        
        successes = 0
        total_generations = 0
        total_time = 0
        total_beneficial = 0
        total_neutral = 0
        
        for trial in range(num_trials):
            if verbose:
                print(f"\nTrial {trial+1}/{num_trials} for n={n}")
            
            # Create target function
            if function_class.__name__ == 'Majority':
                target = function_class(n, relevant_vars=set(range(n)))
            else:
                target = function_class(n)
            
            # Create a random initial hypothesis
            if function_class.__name__ == 'Majority':
                random_vars = set(i for i in range(n) if np.random.random() < 0.5)
                initial_hypothesis = function_class(n, relevant_vars=random_vars)
            else:
                initial_hypothesis = function_class(n)
            
            # Create environment with the specified distribution
            env = Environment(n, target, distribution=distribution, bias=bias)
            
            # Create evolutionary algorithm
            algo = EvolutionaryAlgorithm(
                environment=env,
                initial_hypothesis=initial_hypothesis,
                epsilon=epsilon,
                tolerance=tolerance,
                sample_size=sample_size,
                validation_size=validation_size,
                max_generations=n * 100,  # Scale with problem size
                stagnation_threshold=50
            )
            
            # Run the algorithm
            result = algo.run(verbose=verbose)
            
            # Update statistics
            if result.success:
                successes += 1
            total_generations += result.generations
            total_time += result.elapsed_time
            total_beneficial += result.beneficial_mutations
            total_neutral += result.neutral_mutations
            
            # Save fitness history for plotting
            fitness_histories[n].append(result.fitness_history.copy())
            
            if verbose:
                print(f"Trial {trial+1} completed. Success: {result.success}")
        
        # Calculate statistics for this n
        success_rate = successes / num_trials
        avg_generations = total_generations / num_trials
        avg_time = total_time / num_trials
        avg_beneficial = total_beneficial / num_trials
        avg_neutral = total_neutral / num_trials
        
        # Store results
        results['success_rates'].append(success_rate)
        results['avg_generations'].append(avg_generations)
        results['avg_times'].append(avg_time)
        results['avg_beneficial_mutations'].append(avg_beneficial)
        results['avg_neutral_mutations'].append(avg_neutral)
        
        if verbose:
            print(f"\nResults for n={n}:")
            print(f"Success rate: {success_rate:.2f}")
            print(f"Average generations: {avg_generations:.2f}")
            print(f"Average beneficial mutations: {avg_beneficial:.2f}")
            print(f"Average neutral mutations: {avg_neutral:.2f}")
            print(f"Average time: {avg_time:.2f} seconds")
    
    # Include fitness histories for plotting
    results['fitness_histories'] = fitness_histories
    
    return results


def run_no_neutral_experiment(function_class, n_values: List[int], num_trials: int = 10, 
                              epsilon: float = 0.05, tolerance: float = 0.01, sample_size: int = 1000,
                              validation_size: int = 5000, verbose: bool = False) -> Dict[str, Any]:
    """
    Run an experiment without allowing neutral mutations.
    """
    results = {
        'function_class': function_class.__name__,
        'epsilon': epsilon,
        'tolerance': tolerance,
        'trials': num_trials,
        'n_values': n_values,
        'success_rates': [],
        'avg_generations': [],
        'avg_times': [],
        'avg_beneficial_mutations': [],
        'avg_neutral_mutations': []
    }
    
    # For storing fitness history data for plotting
    fitness_histories = {n: [] for n in n_values}
    
    for n in n_values:
        if verbose:
            print(f"\n--- Running experiments for n={n} ---")
        
        successes = 0
        total_generations = 0
        total_time = 0
        total_beneficial = 0
        total_neutral = 0
        
        for trial in range(num_trials):
            if verbose:
                print(f"\nTrial {trial+1}/{num_trials} for n={n}")
            
            # Create target function
            if function_class.__name__ == 'Majority':
                target = function_class(n, relevant_vars=set(range(n)))
            else:
                target = function_class(n)
            
            # Create a random initial hypothesis
            if function_class.__name__ == 'Majority':
                random_vars = set(i for i in range(n) if np.random.random() < 0.5)
                initial_hypothesis = function_class(n, relevant_vars=random_vars)
            else:
                initial_hypothesis = function_class(n)
            
            # Create environment
            env = Environment(n, target)
            
            # Create evolutionary algorithm
            algo = EvolutionaryAlgorithm(
                environment=env,
                initial_hypothesis=initial_hypothesis,
                epsilon=epsilon,
                tolerance=tolerance,
                sample_size=sample_size,
                validation_size=validation_size,
                max_generations=n * 100,  # Scale with problem size
                stagnation_threshold=50,
                allow_neutral_mutations=False  # Disable neutral mutations
            )
            
            # Run the algorithm
            result = algo.run(verbose=verbose)
            
            # Update statistics
            if result.success:
                successes += 1
            total_generations += result.generations
            total_time += result.elapsed_time
            total_beneficial += result.beneficial_mutations
            total_neutral += result.neutral_mutations
            
            # Save fitness history for plotting
            fitness_histories[n].append(result.fitness_history.copy())
            
            if verbose:
                print(f"Trial {trial+1} completed. Success: {result.success}")
        
        # Calculate statistics for this n
        success_rate = successes / num_trials
        avg_generations = total_generations / num_trials
        avg_time = total_time / num_trials
        avg_beneficial = total_beneficial / num_trials
        avg_neutral = total_neutral / num_trials
        
        # Store results
        results['success_rates'].append(success_rate)
        results['avg_generations'].append(avg_generations)
        results['avg_times'].append(avg_time)
        results['avg_beneficial_mutations'].append(avg_beneficial)
        results['avg_neutral_mutations'].append(avg_neutral)
        
        if verbose:
            print(f"\nResults for n={n}:")
            print(f"Success rate: {success_rate:.2f}")
            print(f"Average generations: {avg_generations:.2f}")
            print(f"Average beneficial mutations: {avg_beneficial:.2f}")
            print(f"Average neutral mutations: {avg_neutral:.2f}")
            print(f"Average time: {avg_time:.2f} seconds")
    
    # Include fitness histories for plotting
    results['fitness_histories'] = fitness_histories
    
    return results


def create_smart_initial_hypothesis(function_class, n: int, target_function) -> BooleanFunction:
    """
    Create a smart initial hypothesis based on the target function class.
    
    This creates an initial hypothesis that's closer to the target function,
    but not identical - giving the algorithm a head start.
    """
    if function_class.__name__ == 'MonotoneConjunction':
        # For conjunction, start with fewer variables than the target might have
        included_vars = set(i for i in range(n) if i % 3 == 0)  # Include every third variable
        return function_class(n, included_vars)
        
    elif function_class.__name__ == 'MonotoneDisjunction':
        # For disjunction, start with more variables than the target might have
        included_vars = set(i for i in range(n) if i % 2 == 0)  # Include every other variable
        return function_class(n, included_vars)
        
    elif function_class.__name__ == 'GeneralConjunction':
        # For general conjunction, start with some variables, mix of positive and negative
        literals = {}
        for i in range(n):
            if i % 3 == 0:
                literals[i] = True  # Positive literal
            elif i % 3 == 1:
                literals[i] = False  # Negative literal
        return function_class(n, literals)
        
    elif function_class.__name__ == 'GeneralDisjunction':
        # For general disjunction, similar approach
        literals = {}
        for i in range(n):
            if i % 2 == 0:
                literals[i] = True  # Positive literal
        return function_class(n, literals)
        
    elif function_class.__name__ == 'Parity':
        # For parity, start with a small set of variables
        included_vars = set(range(min(5, n)))  # First few variables
        return function_class(n, included_vars)
        
    elif function_class.__name__ == 'Majority':
        # For majority, start with about half the variables
        relevant_vars = set(i for i in range(n) if i % 2 == 0)
        return function_class(n, relevant_vars)
    
    # Default case - just use random initialization
    return function_class(n)


def run_smart_init_experiment(function_class, n_values: List[int], num_trials: int = 10, 
                              epsilon: float = 0.05, tolerance: float = 0.01, sample_size: int = 1000,
                              validation_size: int = 5000, verbose: bool = False) -> Dict[str, Any]:
    """
    Run an experiment with smartly pre-selected initial hypotheses.
    """
    results = {
        'function_class': function_class.__name__,
        'epsilon': epsilon,
        'tolerance': tolerance,
        'trials': num_trials,
        'n_values': n_values,
        'success_rates': [],
        'avg_generations': [],
        'avg_times': [],
        'avg_beneficial_mutations': [],
        'avg_neutral_mutations': []
    }
    
    # For storing fitness history data for plotting
    fitness_histories = {n: [] for n in n_values}
    
    for n in n_values:
        if verbose:
            print(f"\n--- Running experiments for n={n} ---")
        
        successes = 0
        total_generations = 0
        total_time = 0
        total_beneficial = 0
        total_neutral = 0
        
        for trial in range(num_trials):
            if verbose:
                print(f"\nTrial {trial+1}/{num_trials} for n={n}")
            
            # Create target function
            if function_class.__name__ == 'Majority':
                target = function_class(n, relevant_vars=set(range(n)))
            else:
                target = function_class(n)
            
            # Create a smart initial hypothesis
            initial_hypothesis = create_smart_initial_hypothesis(function_class, n, target)
            
            # Create environment
            env = Environment(n, target)
            
            # Create evolutionary algorithm
            algo = EvolutionaryAlgorithm(
                environment=env,
                initial_hypothesis=initial_hypothesis,
                epsilon=epsilon,
                tolerance=tolerance,
                sample_size=sample_size,
                validation_size=validation_size,
                max_generations=n * 100,  # Scale with problem size
                stagnation_threshold=50
            )
            
            # Run the algorithm
            result = algo.run(verbose=verbose)
            
            # Update statistics
            if result.success:
                successes += 1
            total_generations += result.generations
            total_time += result.elapsed_time
            total_beneficial += result.beneficial_mutations
            total_neutral += result.neutral_mutations
            
            # Save fitness history for plotting
            fitness_histories[n].append(result.fitness_history.copy())
            
            if verbose:
                print(f"Trial {trial+1} completed. Success: {result.success}")
        
        # Calculate statistics for this n
        success_rate = successes / num_trials
        avg_generations = total_generations / num_trials
        avg_time = total_time / num_trials
        avg_beneficial = total_beneficial / num_trials
        avg_neutral = total_neutral / num_trials
        
        # Store results
        results['success_rates'].append(success_rate)
        results['avg_generations'].append(avg_generations)
        results['avg_times'].append(avg_time)
        results['avg_beneficial_mutations'].append(avg_beneficial)
        results['avg_neutral_mutations'].append(avg_neutral)
        
        if verbose:
            print(f"\nResults for n={n}:")
            print(f"Success rate: {success_rate:.2f}")
            print(f"Average generations: {avg_generations:.2f}")
            print(f"Average beneficial mutations: {avg_beneficial:.2f}")
            print(f"Average neutral mutations: {avg_neutral:.2f}")
            print(f"Average time: {avg_time:.2f} seconds")
    
    # Include fitness histories for plotting
    results['fitness_histories'] = fitness_histories
    
    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run comprehensive evolvability experiments')
    
    parser.add_argument(
        '--n-values', 
        nargs='+',
        type=int,
        default=DEFAULT_N_VALUES,
        help='List of input sizes to test'
    )
    
    parser.add_argument(
        '--trials', 
        type=int,
        default=DEFAULT_TRIALS,
        help='Number of trials for each configuration'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Print verbose output'
    )
    
    parser.add_argument(
        '--skip-standard', 
        action='store_true',
        help='Skip standard tests'
    )
    
    parser.add_argument(
        '--skip-distribution', 
        action='store_true',
        help='Skip distribution tests'
    )
    
    parser.add_argument(
        '--skip-no-neutral', 
        action='store_true',
        help='Skip no-neutral-mutation tests'
    )
    
    parser.add_argument(
        '--skip-smart-init', 
        action='store_true',
        help='Skip smart initialization tests'
    )
    
    return parser.parse_args()


def main():
    """Main function to run all experiments."""
    args = parse_args()
    
    start_time = time.time()
    
    # Create results directory
    results_dir = create_results_dir("full_experiments")
    
    # Run each experiment suite based on command-line flags
    if not args.skip_standard:
        run_standard_tests(results_dir, args.n_values, args.trials, args.verbose)
    
    if not args.skip_distribution:
        run_distribution_tests(results_dir, args.n_values, args.trials, args.verbose)
    
    if not args.skip_no_neutral:
        run_no_neutral_tests(results_dir, args.n_values, args.trials, args.verbose)
    
    if not args.skip_smart_init:
        run_smart_init_tests(results_dir, args.n_values, args.trials, args.verbose)
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")
    print(f"Results saved to {results_dir}")


if __name__ == "__main__":
    main() 