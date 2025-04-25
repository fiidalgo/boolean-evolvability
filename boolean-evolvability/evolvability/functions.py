import numpy as np
from typing import List, Set, Optional
from abc import ABC, abstractmethod


class BooleanFunction(ABC):
    """
    Abstract base class for Boolean functions.
    """
    
    def __init__(self, n: int):
        """
        Initialize a Boolean function with n input variables.
        
        Args:
            n: Number of input variables
        """
        self.n = n
    
    @abstractmethod
    def evaluate(self, x) -> int:
        """
        Evaluate the function on an input.
        
        Args:
            x: Input binary vector of length n
            
        Returns:
            0 or 1, the function output
        """
        pass
    
    @abstractmethod
    def mutate(self) -> List:
        """
        Generate all possible mutations (neighbors) of this function.
        
        Returns:
            List of mutated functions
        """
        pass
    

class MonotoneConjunction(BooleanFunction):
    """
    Monotone conjunction: AND of a subset of variables.
    f(x) = x_i1 ∧ x_i2 ∧ ... ∧ x_ik
    """
    
    def __init__(self, n: int, included_vars: Optional[Set[int]] = None):
        """
        Initialize a monotone conjunction.
        
        Args:
            n: Number of input variables
            included_vars: Set of indices (0-based) of variables included in the conjunction.
                If None, a random subset is selected.
        """
        super().__init__(n)
        
        if included_vars is None:
            # Randomly include each variable with 50% probability
            self.included_vars = set(i for i in range(n) if np.random.random() < 0.5)
        else:
            self.included_vars = included_vars
    
    def evaluate(self, x) -> int:
        """
        Evaluate the conjunction on input x.
        Returns 1 if all included variables are 1, otherwise 0.
        
        Args:
            x: Input binary vector of length n
            
        Returns:
            0 or 1
        """
        # For an empty conjunction (no variables), return 1 (TRUE)
        if not self.included_vars:
            return 1
        
        # Check that all included variables are 1
        for i in self.included_vars:
            if x[i] == 0:
                return 0
        return 1
    
    def mutate(self) -> List['MonotoneConjunction']:
        """
        Generate all possible single-bit mutations of this conjunction.
        Each mutation either adds or removes one variable.
        
        Returns:
            List of mutated conjunctions
        """
        mutations = []
        
        # For each variable currently in the conjunction, create a mutation that removes it
        for i in self.included_vars:
            new_vars = self.included_vars.copy()
            new_vars.remove(i)
            mutations.append(MonotoneConjunction(self.n, new_vars))
        
        # For each variable not in the conjunction, create a mutation that adds it
        for i in range(self.n):
            if i not in self.included_vars:
                new_vars = self.included_vars.copy()
                new_vars.add(i)
                mutations.append(MonotoneConjunction(self.n, new_vars))
        
        return mutations
    
    def __str__(self) -> str:
        """String representation of the conjunction."""
        if not self.included_vars:
            return "TRUE"
        
        terms = [f"x{i+1}" for i in sorted(self.included_vars)]
        return " ∧ ".join(terms)


class MonotoneDisjunction(BooleanFunction):
    """
    Monotone disjunction: OR of a subset of variables.
    f(x) = x_i1 ∨ x_i2 ∨ ... ∨ x_ik
    """
    
    def __init__(self, n: int, included_vars: Optional[Set[int]] = None):
        """
        Initialize a monotone disjunction.
        
        Args:
            n: Number of input variables
            included_vars: Set of indices (0-based) of variables included in the disjunction.
                If None, a random subset is selected.
        """
        super().__init__(n)
        
        if included_vars is None:
            # Randomly include each variable with 50% probability
            self.included_vars = set(i for i in range(n) if np.random.random() < 0.5)
        else:
            self.included_vars = included_vars
    
    def evaluate(self, x) -> int:
        """
        Evaluate the disjunction on input x.
        Returns 1 if any included variable is 1, otherwise 0.
        
        Args:
            x: Input binary vector of length n
            
        Returns:
            0 or 1
        """
        # For an empty disjunction (no variables), return 0 (FALSE)
        if not self.included_vars:
            return 0
        
        # Check if any included variable is 1
        for i in self.included_vars:
            if x[i] == 1:
                return 1
        return 0
    
    def mutate(self) -> List['MonotoneDisjunction']:
        """
        Generate all possible single-bit mutations of this disjunction.
        Each mutation either adds or removes one variable.
        
        Returns:
            List of mutated disjunctions
        """
        mutations = []
        
        # For each variable currently in the disjunction, create a mutation that removes it
        for i in self.included_vars:
            new_vars = self.included_vars.copy()
            new_vars.remove(i)
            mutations.append(MonotoneDisjunction(self.n, new_vars))
        
        # For each variable not in the disjunction, create a mutation that adds it
        for i in range(self.n):
            if i not in self.included_vars:
                new_vars = self.included_vars.copy()
                new_vars.add(i)
                mutations.append(MonotoneDisjunction(self.n, new_vars))
        
        return mutations
    
    def __str__(self) -> str:
        """String representation of the disjunction."""
        if not self.included_vars:
            return "FALSE"
        
        terms = [f"x{i+1}" for i in sorted(self.included_vars)]
        return " ∨ ".join(terms)


class Parity(BooleanFunction):
    """
    Parity function: XOR of a subset of variables.
    f(x) = x_i1 ⊕ x_i2 ⊕ ... ⊕ x_ik
    """
    
    def __init__(self, n: int, included_vars: Optional[Set[int]] = None):
        """
        Initialize a parity function.
        
        Args:
            n: Number of input variables
            included_vars: Set of indices (0-based) of variables included in the parity.
                If None, a random subset is selected.
        """
        super().__init__(n)
        
        if included_vars is None:
            # Randomly include each variable with 50% probability
            self.included_vars = set(i for i in range(n) if np.random.random() < 0.5)
        else:
            self.included_vars = included_vars
    
    def evaluate(self, x) -> int:
        """
        Evaluate the parity function on input x.
        Returns 1 if an odd number of included variables are 1, otherwise 0.
        
        Args:
            x: Input binary vector of length n
            
        Returns:
            0 or 1
        """
        # For an empty parity (no variables), return 0
        if not self.included_vars:
            return 0
        
        # Count how many included variables are 1
        count = 0
        for i in self.included_vars:
            if x[i] == 1:
                count += 1
        
        # Return 1 if count is odd, 0 if count is even
        return count % 2
    
    def mutate(self) -> List['Parity']:
        """
        Generate all possible single-bit mutations of this parity function.
        Each mutation either adds or removes one variable.
        
        Returns:
            List of mutated parity functions
        """
        mutations = []
        
        # For each variable currently in the parity, create a mutation that removes it
        for i in self.included_vars:
            new_vars = self.included_vars.copy()
            new_vars.remove(i)
            mutations.append(Parity(self.n, new_vars))
        
        # For each variable not in the parity, create a mutation that adds it
        for i in range(self.n):
            if i not in self.included_vars:
                new_vars = self.included_vars.copy()
                new_vars.add(i)
                mutations.append(Parity(self.n, new_vars))
        
        return mutations
    
    def __str__(self) -> str:
        """String representation of the parity function."""
        if not self.included_vars:
            return "0"
        
        terms = [f"x{i+1}" for i in sorted(self.included_vars)]
        return " ⊕ ".join(terms)


class Majority(BooleanFunction):
    """
    Majority function: Returns 1 if more than half of the inputs are 1.
    """
    
    def __init__(self, n: int):
        """
        Initialize a majority function.
        
        Args:
            n: Number of input variables
        """
        super().__init__(n)
        # Threshold is set to simple majority (n/2 rounded up)
        self.threshold = (n + 1) // 2
    
    def evaluate(self, x) -> int:
        """
        Evaluate the majority function on input x.
        Returns 1 if more than half of the inputs are 1, otherwise 0.
        
        Args:
            x: Input binary vector of length n
            
        Returns:
            0 or 1
        """
        return 1 if np.sum(x) >= self.threshold else 0
    
    def mutate(self) -> List['Majority']:
        """
        For the majority function, mutations don't make much sense
        in the standard form. We could change the threshold, but
        that would go against the definition of majority.
        For simplicity, we return an empty list here.
        
        Returns:
            Empty list
        """
        return []
    
    def __str__(self) -> str:
        """String representation of the majority function."""
        return f"MAJ({self.n})" 