import numpy as np
from typing import List, Set, Optional, Tuple, Dict
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


class GeneralConjunction(BooleanFunction):
    """
    General conjunction: AND of a subset of variables and their negations.
    f(x) = l_1 ∧ l_2 ∧ ... ∧ l_k
    where l_i is either x_i or ¬x_i
    """
    
    def __init__(self, n: int, literals: Optional[Dict[int, bool]] = None):
        """
        Initialize a general conjunction.
        
        Args:
            n: Number of input variables
            literals: Dictionary mapping variable indices to whether they appear in non-negated form.
                     Key=index, Value=True if literal is positive (x_i), False if negated (¬x_i)
                     If None, a random subset with random polarities is selected.
        """
        super().__init__(n)
        
        if literals is None:
            # Randomly include each variable with 50% probability and random polarity
            self.literals = {}
            for i in range(n):
                if np.random.random() < 0.5:  # 50% chance to include this variable
                    self.literals[i] = np.random.random() < 0.5  # 50% chance to be positive/negative
        else:
            self.literals = literals
    
    def evaluate(self, x) -> int:
        """
        Evaluate the general conjunction on input x.
        Returns 1 if all literals evaluate to 1, otherwise 0.
        
        Args:
            x: Input binary vector of length n
            
        Returns:
            0 or 1
        """
        # For an empty conjunction (no literals), return 1 (TRUE)
        if not self.literals:
            return 1
        
        # Check all literals
        for index, is_positive in self.literals.items():
            if is_positive:  # x_i
                if x[index] == 0:
                    return 0
            else:  # ¬x_i
                if x[index] == 1:
                    return 0
        return 1
    
    def mutate(self) -> List['GeneralConjunction']:
        """
        Generate all possible single-literal mutations of this general conjunction.
        Mutations: add a new literal, remove a literal, or flip a literal's polarity.
        
        Returns:
            List of mutated general conjunctions
        """
        mutations = []
        
        # 1. Remove each existing literal
        for index in self.literals:
            new_literals = self.literals.copy()
            del new_literals[index]
            mutations.append(GeneralConjunction(self.n, new_literals))
        
        # 2. Flip each literal's polarity
        for index, is_positive in self.literals.items():
            new_literals = self.literals.copy()
            new_literals[index] = not is_positive
            mutations.append(GeneralConjunction(self.n, new_literals))
        
        # 3. Add each possible new literal (both positive and negative)
        for i in range(self.n):
            if i not in self.literals:
                # Add positive literal (x_i)
                new_literals_pos = self.literals.copy()
                new_literals_pos[i] = True
                mutations.append(GeneralConjunction(self.n, new_literals_pos))
                
                # Add negative literal (¬x_i)
                new_literals_neg = self.literals.copy()
                new_literals_neg[i] = False
                mutations.append(GeneralConjunction(self.n, new_literals_neg))
        
        return mutations
    
    def __str__(self) -> str:
        """String representation of the general conjunction."""
        if not self.literals:
            return "TRUE"
        
        terms = []
        for index, is_positive in sorted(self.literals.items()):
            if is_positive:
                terms.append(f"x{index+1}")
            else:
                terms.append(f"¬x{index+1}")
        return " ∧ ".join(terms)


class GeneralDisjunction(BooleanFunction):
    """
    General disjunction: OR of a subset of variables and their negations.
    f(x) = l_1 ∨ l_2 ∨ ... ∨ l_k
    where l_i is either x_i or ¬x_i
    """
    
    def __init__(self, n: int, literals: Optional[Dict[int, bool]] = None):
        """
        Initialize a general disjunction.
        
        Args:
            n: Number of input variables
            literals: Dictionary mapping variable indices to whether they appear in non-negated form.
                     Key=index, Value=True if literal is positive (x_i), False if negated (¬x_i)
                     If None, a random subset with random polarities is selected.
        """
        super().__init__(n)
        
        if literals is None:
            # Randomly include each variable with 50% probability and random polarity
            self.literals = {}
            for i in range(n):
                if np.random.random() < 0.5:  # 50% chance to include this variable
                    self.literals[i] = np.random.random() < 0.5  # 50% chance to be positive/negative
        else:
            self.literals = literals
    
    def evaluate(self, x) -> int:
        """
        Evaluate the general disjunction on input x.
        Returns 1 if at least one literal evaluates to 1, otherwise 0.
        
        Args:
            x: Input binary vector of length n
            
        Returns:
            0 or 1
        """
        # For an empty disjunction (no literals), return 0 (FALSE)
        if not self.literals:
            return 0
        
        # Check all literals
        for index, is_positive in self.literals.items():
            if is_positive:  # x_i
                if x[index] == 1:
                    return 1
            else:  # ¬x_i
                if x[index] == 0:
                    return 1
        return 0
    
    def mutate(self) -> List['GeneralDisjunction']:
        """
        Generate all possible single-literal mutations of this general disjunction.
        Mutations: add a new literal, remove a literal, or flip a literal's polarity.
        
        Returns:
            List of mutated general disjunctions
        """
        mutations = []
        
        # 1. Remove each existing literal
        for index in self.literals:
            new_literals = self.literals.copy()
            del new_literals[index]
            mutations.append(GeneralDisjunction(self.n, new_literals))
        
        # 2. Flip each literal's polarity
        for index, is_positive in self.literals.items():
            new_literals = self.literals.copy()
            new_literals[index] = not is_positive
            mutations.append(GeneralDisjunction(self.n, new_literals))
        
        # 3. Add each possible new literal (both positive and negative)
        for i in range(self.n):
            if i not in self.literals:
                # Add positive literal (x_i)
                new_literals_pos = self.literals.copy()
                new_literals_pos[i] = True
                mutations.append(GeneralDisjunction(self.n, new_literals_pos))
                
                # Add negative literal (¬x_i)
                new_literals_neg = self.literals.copy()
                new_literals_neg[i] = False
                mutations.append(GeneralDisjunction(self.n, new_literals_neg))
        
        return mutations
    
    def __str__(self) -> str:
        """String representation of the general disjunction."""
        if not self.literals:
            return "FALSE"
        
        terms = []
        for index, is_positive in sorted(self.literals.items()):
            if is_positive:
                terms.append(f"x{index+1}")
            else:
                terms.append(f"¬x{index+1}")
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
    Majority function: Returns 1 if at least threshold of the inputs are 1.
    """
    
    def __init__(self, n: int, threshold: Optional[int] = None):
        """
        Initialize a majority function.
        
        Args:
            n: Number of input variables
            threshold: Threshold value. If None, will be set to a random value in [0,n]
        """
        super().__init__(n)
        # If no threshold provided, select a random value between 0 and n (inclusive)
        if threshold is None:
            self.threshold = np.random.randint(0, n+1)
        else:
            self.threshold = threshold
    
    def evaluate(self, x) -> int:
        """
        Evaluate the majority function on input x.
        Returns 1 if at least threshold inputs are 1, otherwise 0.
        
        Args:
            x: Input binary vector of length n
            
        Returns:
            0 or 1
        """
        return 1 if np.sum(x) >= self.threshold else 0
    
    def mutate(self) -> List['Majority']:
        """
        Generate mutations of the threshold function by adjusting the threshold.
        Mutations can increase or decrease the threshold by 1, within valid bounds.
        
        Returns:
            List of mutated majority functions
        """
        mutations = []
        
        # Add a mutation that increases the threshold (if possible)
        if self.threshold < self.n:
            mutations.append(Majority(self.n, self.threshold + 1))
        
        # Add a mutation that decreases the threshold (if possible)
        if self.threshold > 0:
            mutations.append(Majority(self.n, self.threshold - 1))
        
        return mutations
    
    def __str__(self) -> str:
        """String representation of the majority function."""
        return f"Threshold({self.n}, t={self.threshold})" 