#!/usr/bin/env python3
"""
Simple tests to verify that the evolvability simulator is working correctly.
"""

import numpy as np
from evolvability.functions import MonotoneConjunction, MonotoneDisjunction, Parity, Majority
from evolvability.environment import Environment
from evolvability.evolve import EvolutionaryAlgorithm


def test_function_evaluation():
    """Test that Boolean functions evaluate correctly."""
    print("Testing function evaluation...")
    
    # Test MonotoneConjunction
    n = 5
    included_vars = {0, 2}  # x1 AND x3
    conj = MonotoneConjunction(n, included_vars)
    
    # Test cases
    test_inputs = [
        np.array([1, 0, 1, 0, 0]),  # x1=1, x3=1 -> should return 1
        np.array([1, 1, 0, 1, 1]),  # x1=1, x3=0 -> should return 0
        np.array([0, 1, 1, 1, 1]),  # x1=0, x3=1 -> should return 0
    ]
    
    expected_outputs = [1, 0, 0]
    
    for i, test_input in enumerate(test_inputs):
        output = conj.evaluate(test_input)
        print(f"  Input: {test_input}, Expected: {expected_outputs[i]}, Got: {output}")
        assert output == expected_outputs[i], f"MonotoneConjunction test failed for input {test_input}"
    
    # Test MonotoneDisjunction
    included_vars = {1, 3}  # x2 OR x4
    disj = MonotoneDisjunction(n, included_vars)
    
    test_inputs = [
        np.array([0, 1, 0, 0, 0]),  # x2=1, x4=0 -> should return 1
        np.array([0, 0, 0, 1, 0]),  # x2=0, x4=1 -> should return 1
        np.array([0, 0, 0, 0, 1]),  # x2=0, x4=0 -> should return 0
    ]
    
    expected_outputs = [1, 1, 0]
    
    for i, test_input in enumerate(test_inputs):
        output = disj.evaluate(test_input)
        print(f"  Input: {test_input}, Expected: {expected_outputs[i]}, Got: {output}")
        assert output == expected_outputs[i], f"MonotoneDisjunction test failed for input {test_input}"
    
    # Test Parity
    included_vars = {0, 1, 3}  # x1 XOR x2 XOR x4
    parity = Parity(n, included_vars)
    
    test_inputs = [
        np.array([1, 0, 0, 0, 0]),  # Only x1=1 -> odd count -> should return 1
        np.array([1, 1, 0, 0, 0]),  # x1=1, x2=1 -> even count -> should return 0
        np.array([1, 1, 0, 1, 0]),  # x1=1, x2=1, x4=1 -> odd count -> should return 1
    ]
    
    expected_outputs = [1, 0, 1]
    
    for i, test_input in enumerate(test_inputs):
        output = parity.evaluate(test_input)
        print(f"  Input: {test_input}, Expected: {expected_outputs[i]}, Got: {output}")
        assert output == expected_outputs[i], f"Parity test failed for input {test_input}"
    
    # Test Majority
    majority = Majority(5)  # Majority of 5 variables (threshold = 3)
    
    test_inputs = [
        np.array([1, 1, 1, 0, 0]),  # 3 ones -> should return 1
        np.array([1, 1, 0, 0, 0]),  # 2 ones -> should return 0
        np.array([1, 1, 1, 1, 0]),  # 4 ones -> should return 1
    ]
    
    expected_outputs = [1, 0, 1]
    
    for i, test_input in enumerate(test_inputs):
        output = majority.evaluate(test_input)
        print(f"  Input: {test_input}, Expected: {expected_outputs[i]}, Got: {output}")
        assert output == expected_outputs[i], f"Majority test failed for input {test_input}"
    
    print("All function evaluation tests passed!\n")


def test_mutation():
    """Test that mutation operators work correctly."""
    print("Testing mutation operators...")
    
    # Test MonotoneConjunction mutation
    n = 3
    included_vars = {0, 1}  # x1 AND x2
    conj = MonotoneConjunction(n, included_vars)
    
    mutations = conj.mutate()
    print(f"  Original conjunction: {conj}")
    print(f"  Number of mutations: {len(mutations)}")
    for i, mutation in enumerate(mutations):
        print(f"  Mutation {i+1}: {mutation}")
    
    # We expect n mutations (3 in this case)
    # - Remove x1
    # - Remove x2
    # - Add x3
    assert len(mutations) == n, f"Expected {n} mutations, got {len(mutations)}"
    
    # Test MonotoneDisjunction mutation
    included_vars = {0}  # x1
    disj = MonotoneDisjunction(n, included_vars)
    
    mutations = disj.mutate()
    print(f"\n  Original disjunction: {disj}")
    print(f"  Number of mutations: {len(mutations)}")
    for i, mutation in enumerate(mutations):
        print(f"  Mutation {i+1}: {mutation}")
    
    # We expect n mutations (3 in this case)
    # - Remove x1
    # - Add x2
    # - Add x3
    assert len(mutations) == n, f"Expected {n} mutations, got {len(mutations)}"
    
    print("All mutation tests passed!\n")


def test_environment():
    """Test the environment's sample generation and fitness evaluation."""
    print("Testing environment...")
    
    n = 5
    
    # Create a target conjunction x1 AND x3
    target = MonotoneConjunction(n, {0, 2})
    
    # Create environment
    env = Environment(n, target)
    
    # Generate a sample
    sample_size = 10
    samples = env.draw_sample(sample_size)
    
    print(f"  Generated {len(samples)} samples")
    for i, (x, y) in enumerate(samples[:3]):  # Print first 3 samples
        print(f"  Sample {i+1}: Input={x}, Label={y}")
    
    # Test fitness evaluation
    # 1. Create a hypothesis identical to target (should have perfect fitness)
    perfect_hypothesis = MonotoneConjunction(n, {0, 2})
    perfect_fitness = env.evaluate_fitness(perfect_hypothesis, 1000)
    print(f"  Perfect hypothesis fitness: {perfect_fitness}")
    assert abs(perfect_fitness - 1.0) < 0.1, "Perfect hypothesis should have fitness close to 1.0"
    
    # 2. Create a completely wrong hypothesis (should have poor fitness)
    wrong_hypothesis = MonotoneConjunction(n, {1, 3, 4})
    wrong_fitness = env.evaluate_fitness(wrong_hypothesis, 1000)
    print(f"  Wrong hypothesis fitness: {wrong_fitness}")
    assert wrong_fitness < 0.9, "Wrong hypothesis should have fitness significantly below 1.0"
    
    print("All environment tests passed!\n")


def test_evolution_simple():
    """Run a very simple evolution test."""
    print("Running a simple evolution test...")
    
    n = 5
    
    # Create a target conjunction x1 AND x3
    target_vars = {0, 2}
    target = MonotoneConjunction(n, target_vars)
    print(f"  Target function: {target}")
    
    # Create an initial hypothesis with all variables
    initial_vars = set(range(n))
    initial_hypothesis = MonotoneConjunction(n, initial_vars)
    print(f"  Initial hypothesis: {initial_hypothesis}")
    
    # Create environment and algorithm
    env = Environment(n, target)
    algo = EvolutionaryAlgorithm(
        environment=env,
        initial_hypothesis=initial_hypothesis,
        epsilon=0.05,
        sample_size=1000,
        max_generations=100,
        stagnation_threshold=20
    )
    
    # Run the algorithm
    print("  Running evolution...")
    result = algo.run(verbose=False)
    
    # Print results
    print("\n  Results:")
    print(f"  Success: {result.success}")
    print(f"  Generations: {result.generations}")
    print(f"  Final fitness: {result.final_fitness:.4f}")
    print(f"  Final hypothesis: {result.final_hypothesis}")
    
    # For a conjunction, we expect evolution to succeed
    assert result.success, "Evolution of conjunction should succeed"
    
    print("Simple evolution test passed!")


if __name__ == "__main__":
    # Run tests
    test_function_evaluation()
    test_mutation()
    test_environment()
    test_evolution_simple() 