import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import time
from dataclasses import dataclass
from evolvability.environment import Environment
from evolvability.functions import BooleanFunction


@dataclass
class EvolutionResult:
    """
    Data class to store results of an evolutionary run.
    """
    success: bool  # Whether evolution was successful
    generations: int  # Number of generations
    final_hypothesis: BooleanFunction  # Final evolved hypothesis
    final_fitness: float  # Final fitness
    fitness_history: List[float]  # Fitness values throughout evolution
    elapsed_time: float  # Time taken for evolution
    beneficial_mutations: int  # Count of beneficial mutations accepted
    neutral_mutations: int  # Count of neutral mutations accepted
    mutation_history: List[str]  # History of mutation types (beneficial/neutral/none)


class EvolutionaryAlgorithm:
    """
    Implementation of Valiant's evolvability model.
    """
    
    def __init__(self, 
                 environment: Environment,
                 initial_hypothesis: BooleanFunction,
                 epsilon: float = 0.05,
                 tolerance: float = 0.01,
                 sample_size: int = 1000,
                 validation_size: int = 5000,
                 max_generations: int = 1000,
                 stagnation_threshold: int = 50,
                 allow_neutral_mutations: bool = True):
        """
        Initialize the evolutionary algorithm.
        
        Args:
            environment: The environment that provides examples and evaluates fitness
            initial_hypothesis: The starting hypothesis
            epsilon: Target error threshold (evolution succeeds if error < epsilon)
            tolerance: Tolerance parameter (t) for determining beneficial/neutral mutations
            sample_size: Number of examples to use for fitness evaluation during evolution
            validation_size: Number of examples to use for final validation
            max_generations: Maximum number of generations to run
            stagnation_threshold: Number of generations with no accepted mutation before stopping
            allow_neutral_mutations: Whether to allow neutral mutations or only beneficial ones
        """
        self.environment = environment
        self.current_hypothesis = initial_hypothesis
        self.epsilon = epsilon
        self.tolerance = tolerance
        self.sample_size = sample_size
        self.validation_size = validation_size
        self.max_generations = max_generations
        self.stagnation_threshold = stagnation_threshold
        self.allow_neutral_mutations = allow_neutral_mutations
        
        # Mutation statistics
        self.beneficial_mutations = 0
        self.neutral_mutations = 0
        self.mutation_history = []
        
        # Validation
        if not isinstance(initial_hypothesis.n, int) or initial_hypothesis.n <= 0:
            raise ValueError("Number of variables must be a positive integer")
        if not (0 < epsilon < 1):
            raise ValueError("Epsilon must be between 0 and 1")
        if not (0 < tolerance < 1):
            raise ValueError("Tolerance must be between 0 and 1")
        if not isinstance(sample_size, int) or sample_size <= 0:
            raise ValueError("Sample size must be a positive integer")
    
    def run(self, verbose: bool = False) -> EvolutionResult:
        """
        Run the evolutionary algorithm until a stopping condition is met.
        
        Args:
            verbose: If True, print progress information
            
        Returns:
            EvolutionResult object containing results of the run
        """
        start_time = time.time()
        
        # Initialize tracking variables
        generation = 0
        stagnation_counter = 0
        fitness_history = []
        
        # Evaluate initial hypothesis
        current_fitness = self.environment.evaluate_fitness(
            self.current_hypothesis, self.sample_size)
        fitness_history.append(current_fitness)
        
        if verbose:
            print(f"Generation 0: Fitness = {current_fitness:.4f}")
        
        # Main evolutionary loop
        while generation < self.max_generations:
            generation += 1
            
            # Generate mutations (neighbors)
            candidates = self.current_hypothesis.mutate()
            
            # If no mutations are possible, break
            if not candidates:
                if verbose:
                    print(f"No mutations possible. Ending evolution.")
                break
            
            # Evaluate all candidates and find the best
            best_fitness = current_fitness  # Start with current fitness (allow neutrality)
            best_hypothesis = self.current_hypothesis
            improvement_found = False
            accepted_neutral = False
            
            for candidate in candidates:
                candidate_fitness = self.environment.evaluate_fitness(
                    candidate, self.sample_size)
                
                # Select candidate if it's better by at least tolerance
                if candidate_fitness >= best_fitness + self.tolerance:
                    best_fitness = candidate_fitness
                    best_hypothesis = candidate
                    improvement_found = True
                # Or if it's neutral (not worse than by tolerance) and we decide to accept it
                elif self.allow_neutral_mutations and candidate_fitness >= best_fitness - self.tolerance and candidate is not self.current_hypothesis:
                    # In case of a tie, flip a coin to decide
                    if np.random.random() < 0.5:
                        best_hypothesis = candidate
                        accepted_neutral = True
            
            # Update the current hypothesis if a mutation was accepted
            mutation_occurred = (best_hypothesis is not self.current_hypothesis)
            if mutation_occurred:
                # Record mutation type
                if improvement_found:
                    self.beneficial_mutations += 1
                    self.mutation_history.append("beneficial")
                elif accepted_neutral:
                    self.neutral_mutations += 1
                    self.mutation_history.append("neutral")
                
                # Update hypothesis and fitness
                self.current_hypothesis = best_hypothesis
                current_fitness = best_fitness
                
                # Reset stagnation counter when any mutation is accepted
                stagnation_counter = 0
            else:
                # No mutation was accepted
                self.mutation_history.append("none")
                stagnation_counter += 1
                if stagnation_counter >= self.stagnation_threshold:
                    if verbose:
                        print(f"Stagnation detected after {generation} generations.")
                    break
            
            # Record fitness history
            fitness_history.append(current_fitness)
            
            # Check success condition (fitness >= 1-epsilon)
            if current_fitness >= 1 - self.epsilon:
                if verbose:
                    print(f"Success! Reached target fitness after {generation} generations.")
                break
            
            if verbose and generation % 10 == 0:
                print(f"Generation {generation}: Fitness = {current_fitness:.4f}")
        
        # Final evaluation with a larger sample for more accurate assessment
        final_fitness = self.environment.evaluate_fitness(
            self.current_hypothesis, self.validation_size)
        
        elapsed_time = time.time() - start_time
        
        # Create and return result
        result = EvolutionResult(
            success=(final_fitness >= 1 - self.epsilon),
            generations=generation,
            final_hypothesis=self.current_hypothesis,
            final_fitness=final_fitness,
            fitness_history=fitness_history,
            elapsed_time=elapsed_time,
            beneficial_mutations=self.beneficial_mutations,
            neutral_mutations=self.neutral_mutations,
            mutation_history=self.mutation_history
        )
        
        if verbose:
            print(f"Evolution completed after {generation} generations.")
            print(f"Final fitness: {final_fitness:.4f}")
            print(f"Success: {result.success}")
            print(f"Beneficial mutations: {self.beneficial_mutations}")
            print(f"Neutral mutations: {self.neutral_mutations}")
            print(f"Elapsed time: {elapsed_time:.2f} seconds")
        
        return result


def run_experiment(function_class, n_values: List[int], num_trials: int = 10, 
                   epsilon: float = 0.05, tolerance: float = 0.01, sample_size: int = 1000,
                   validation_size: int = 5000, verbose: bool = False) -> Dict[str, Any]:
    """
    Run a full experiment with multiple trials for different problem sizes.
    
    Args:
        function_class: Class of the target function (e.g., MonotoneConjunction)
        n_values: List of input sizes to test
        num_trials: Number of trials to run for each configuration
        epsilon: Target error threshold
        tolerance: Tolerance parameter (t) for determining beneficial/neutral mutations
        sample_size: Number of examples for fitness evaluation during evolution
        validation_size: Number of examples for final validation
        verbose: Whether to print progress information
        
    Returns:
        Dictionary containing experiment results
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
                # For Majority, use all variables with majority threshold
                target = function_class(n, relevant_vars=set(range(n)))
            else:
                target = function_class(n)
            
            # Create a random initial hypothesis
            if function_class.__name__ == 'Majority':
                # For Majority, start with a random subset of variables
                random_vars = set(i for i in range(n) if np.random.random() < 0.5)
                initial_hypothesis = function_class(n, relevant_vars=random_vars)
            else:
                initial_hypothesis = function_class(n)
            
            # Create environment and evolutionary algorithm
            env = Environment(n, target)
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
            # Pad shorter histories with final value to make them equal length
            padded_history = result.fitness_history.copy()
            fitness_histories[n].append(padded_history)
            
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