import numpy as np
from typing import List, Tuple, Callable, Optional, Union, Dict, Any


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
            distribution: Type of distribution to sample from ('uniform', 'biased', 'binomial', 'beta', etc.)
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
        elif self.distribution == "binomial":
            # Binomial distribution: Each input is a binary vector where number of 1s follows a binomial distribution
            # Default params: p=0.3, trials=min(n, 3)
            p = 0.3  # Probability parameter for binomial
            trials = min(self.n, 3)  # Number of trials parameter
            
            # Generate biased inputs by sampling binomial for each example (controls number of 1s)
            inputs = np.zeros((sample_size, self.n), dtype=int)
            for i in range(sample_size):
                # Number of 1s to include (following binomial distribution)
                num_ones = np.random.binomial(trials, p)
                
                # Randomly place the 1s
                if num_ones > 0:
                    ones_positions = np.random.choice(self.n, size=num_ones, replace=False)
                    inputs[i, ones_positions] = 1
        elif self.distribution == "beta":
            # Beta distribution: probability of each bit being 1 follows a Beta(alpha, beta) distribution
            # Default params: alpha=2, beta=5 (skewed toward 0)
            alpha = 2.0
            beta = 5.0
            
            # Generate a different p for each example from Beta distribution
            inputs = np.zeros((sample_size, self.n), dtype=int)
            for i in range(sample_size):
                # Sample probability from beta distribution
                p = np.random.beta(alpha, beta)
                
                # Generate bits with this probability
                inputs[i] = (np.random.random(size=self.n) < p).astype(int)
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