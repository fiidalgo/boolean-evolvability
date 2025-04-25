import numpy as np
from typing import List, Tuple, Callable, Optional, Union


class Environment:
    """
    Environment class that handles generating random examples according to a distribution
    and evaluating hypotheses against a target function.
    """
    
    def __init__(self, n: int, target_function, distribution: str = "uniform", 
                 bias: Optional[Union[float, List[float]]] = None):
        """
        Initialize the environment.
        
        Args:
            n: Number of input bits
            target_function: The target function to learn
            distribution: Type of distribution to sample from ('uniform', 'biased', etc.)
            bias: Bias parameter(s) for biased distributions
        """
        self.n = n
        self.target_function = target_function
        self.distribution = distribution
        self.bias = bias
        
        # Validate bias parameter based on distribution type
        if distribution == "biased" and bias is None:
            self.bias = 0.5  # Default to 0.5 if not specified
        
    def draw_sample(self, sample_size: int) -> List[Tuple[np.ndarray, int]]:
        """
        Generate a random sample of examples according to the distribution.
        
        Args:
            sample_size: Number of examples to generate
            
        Returns:
            List of tuples (x, y) where x is the input vector and y is the label
        """
        if self.distribution == "uniform":
            # Generate uniform random binary vectors
            inputs = np.random.randint(0, 2, size=(sample_size, self.n))
        elif self.distribution == "biased":
            # Generate biased random binary vectors
            if isinstance(self.bias, float):
                # Same bias for all bits
                inputs = (np.random.random(size=(sample_size, self.n)) < self.bias).astype(int)
            else:
                # Different bias for each bit
                inputs = np.zeros((sample_size, self.n), dtype=int)
                for i in range(self.n):
                    inputs[:, i] = (np.random.random(size=sample_size) < self.bias[i]).astype(int)
        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")
        
        # Compute labels using the target function
        labels = np.array([self.target_function.evaluate(x) for x in inputs])
        
        return list(zip(inputs, labels))
    
    def evaluate_fitness(self, hypothesis, sample_size: int) -> float:
        """
        Evaluate the fitness of a hypothesis by generating random examples
        and calculating the fraction of correct predictions.
        
        Args:
            hypothesis: The hypothesis to evaluate
            sample_size: Number of examples to use for evaluation
            
        Returns:
            Fitness value in [0, 1] representing accuracy
        """
        samples = self.draw_sample(sample_size)
        correct_count = 0
        
        for x, y in samples:
            if hypothesis.evaluate(x) == y:
                correct_count += 1
                
        return correct_count / sample_size 