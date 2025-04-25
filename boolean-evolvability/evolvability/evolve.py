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


class EvolutionaryAlgorithm:
    """
    Implementation of Valiant's evolvability model.
    """
    
    def __init__(self, 
                 environment: Environment,
                 initial_hypothesis: BooleanFunction,
                 epsilon: float = 0.05,
                 sample_size: int = 1000,
                 max_generations: int = 1000,
                 stagnation_threshold: int = 50):
        """
        Initialize the evolutionary algorithm.
        
        Args:
            environment: The environment that provides examples and evaluates fitness
            initial_hypothesis: The starting hypothesis
            epsilon: Target error threshold (evolution succeeds if error < epsilon)
            sample_size: Number of examples to use for fitness evaluation
            max_generations: Maximum number of generations to run
            stagnation_threshold: Number of generations with no improvement before stopping
        """
        self.environment = environment
        self.current_hypothesis = initial_hypothesis
        self.epsilon = epsilon
        self.sample_size = sample_size
        self.max_generations = max_generations
        self.stagnation_threshold = stagnation_threshold
        
        # Validation
        if not isinstance(initial_hypothesis.n, int) or initial_hypothesis.n <= 0:
            raise ValueError("Number of variables must be a positive integer")
        if not (0 < epsilon < 1):
            raise ValueError("Epsilon must be between 0 and 1")
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
            
            for candidate in candidates:
                candidate_fitness = self.environment.evaluate_fitness(
                    candidate, self.sample_size)
                
                # Select candidate if it's better (or equal, allowing neutral drift)
                if candidate_fitness >= best_fitness:
                    # In case of a tie, flip a coin to decide
                    if candidate_fitness == best_fitness and np.random.random() < 0.5:
                        continue
                    best_fitness = candidate_fitness
                    best_hypothesis = candidate
            
            # Update the current hypothesis
            improvement = best_fitness - current_fitness
            self.current_hypothesis = best_hypothesis
            current_fitness = best_fitness
            fitness_history.append(current_fitness)
            
            # Check for stagnation
            if improvement <= 0:
                stagnation_counter += 1
                if stagnation_counter >= self.stagnation_threshold:
                    if verbose:
                        print(f"Stagnation detected after {generation} generations.")
                    break
            else:
                stagnation_counter = 0
            
            # Check success condition (fitness >= 1-epsilon)
            if current_fitness >= 1 - self.epsilon:
                if verbose:
                    print(f"Success! Reached target fitness after {generation} generations.")
                break
            
            if verbose and generation % 10 == 0:
                print(f"Generation {generation}: Fitness = {current_fitness:.4f}")
        
        # Final evaluation with a larger sample for more accurate assessment
        final_fitness = self.environment.evaluate_fitness(
            self.current_hypothesis, self.sample_size * 5)
        
        elapsed_time = time.time() - start_time
        
        # Create and return result
        result = EvolutionResult(
            success=(final_fitness >= 1 - self.epsilon),
            generations=generation,
            final_hypothesis=self.current_hypothesis,
            final_fitness=final_fitness,
            fitness_history=fitness_history,
            elapsed_time=elapsed_time
        )
        
        if verbose:
            print(f"Evolution completed after {generation} generations.")
            print(f"Final fitness: {final_fitness:.4f}")
            print(f"Success: {result.success}")
            print(f"Elapsed time: {elapsed_time:.2f} seconds")
        
        return result


def run_experiment(function_class, n_values: List[int], num_trials: int = 10, 
                   epsilon: float = 0.05, verbose: bool = False) -> Dict[str, Any]:
    """
    Run a full experiment with multiple trials for different problem sizes.
    
    Args:
        function_class: Class of the target function (e.g., MonotoneConjunction)
        n_values: List of input sizes to test
        num_trials: Number of trials to run for each configuration
        epsilon: Target error threshold
        verbose: Whether to print progress information
        
    Returns:
        Dictionary containing experiment results
    """
    results = {
        'function_class': function_class.__name__,
        'epsilon': epsilon,
        'trials': num_trials,
        'n_values': n_values,
        'success_rates': [],
        'avg_generations': [],
        'avg_times': []
    }
    
    for n in n_values:
        if verbose:
            print(f"\n--- Running experiments for n={n} ---")
        
        successes = 0
        total_generations = 0
        total_time = 0
        
        for trial in range(num_trials):
            if verbose:
                print(f"\nTrial {trial+1}/{num_trials} for n={n}")
            
            # Create a random target function
            target = function_class(n)
            
            # Create a random initial hypothesis
            initial_hypothesis = function_class(n)
            
            # Create environment and evolutionary algorithm
            env = Environment(n, target)
            algo = EvolutionaryAlgorithm(
                environment=env,
                initial_hypothesis=initial_hypothesis,
                epsilon=epsilon,
                sample_size=1000,  # Use a reasonable sample size
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
            
            if verbose:
                print(f"Trial {trial+1} completed. Success: {result.success}")
        
        # Calculate statistics for this n
        success_rate = successes / num_trials
        avg_generations = total_generations / num_trials
        avg_time = total_time / num_trials
        
        # Store results
        results['success_rates'].append(success_rate)
        results['avg_generations'].append(avg_generations)
        results['avg_times'].append(avg_time)
        
        if verbose:
            print(f"\nResults for n={n}:")
            print(f"Success rate: {success_rate:.2f}")
            print(f"Average generations: {avg_generations:.2f}")
            print(f"Average time: {avg_time:.2f} seconds")
    
    return results 