# Boolean Evolvability Simulator

This project implements a computational model of Valiant's evolvability framework, as introduced in his 2009 paper "Evolvability". The simulator tests whether different classes of Boolean functions can evolve under the constraints of Darwinian evolution, viewed as a restricted form of PAC learning.

## Background

Leslie Valiant's 2009 paper "Evolvability" introduced a learning-theoretic model of Darwinian evolution, showing that certain classes of Boolean functions can "evolve" with polynomial resources under restricted conditions. In this model:

- Evolution is viewed as a constrained form of PAC learning
- Each generation produces slight random variants (mutations) of the current hypothesis
- Selection is guided only by aggregate performance on random samples
- There is no direct access to individual examples (statistical query model)

The model proves that monotone conjunctions and disjunctions are evolvable under the uniform distribution, while functions like parity are not evolvable in this framework.

## Phase 1 Implementation

This implementation adheres to the original constraints of Valiant's model:

1. **Random example distribution**: Initially uses the uniform distribution for generating examples
2. **Empirical fitness evaluation**: Defines fitness as the fraction of examples on which the hypothesis agrees with the target concept
3. **Restricted mutations**: Only small, local changes are allowed (adding/removing one variable at a time)
4. **Selection based on fitness**: Selection follows statistical query learning principles
5. **Clear stopping criteria**: Evolution succeeds when accuracy reaches 1-ε

### Function Classes Tested

The simulator tests the following Boolean function classes:

- **Monotone Conjunctions**: AND of a subset of variables
- **Monotone Disjunctions**: OR of a subset of variables
- **Parity Functions**: XOR of a subset of variables
- **Majority Functions** (optional): Returns 1 if majority of inputs are 1

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/boolean-evolvability.git
cd boolean-evolvability

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy matplotlib
```

## Usage

### Running a Single Experiment

To run a quick test with a single target function:

```bash
python main.py --single-run --verbose
```

### Running Full Experiments

To run experiments across multiple function classes and problem sizes:

```bash
python main.py --function-classes conjunction disjunction parity --n-values 10 20 50 --trials 10
```

### Command Line Arguments

- `--function-classes`: List of function classes to test (conjunction, disjunction, parity, majority)
- `--n-values`: List of input sizes to test
- `--trials`: Number of trials for each configuration
- `--epsilon`: Target error threshold (default: 0.05)
- `--sample-size`: Number of examples for fitness evaluation (default: 1000)
- `--verbose`: Print detailed progress
- `--single-run`: Run a single experiment for testing

## Results

Results are saved in the `results/` directory with timestamps:

- JSON files with experimental statistics
- PNG plots showing:
  - Fitness history over generations
  - Success rates across problem sizes
  - Comparison of different function classes

## Project Structure

- `evolvability/`: Main package
  - `environment.py`: Handles random example generation and fitness evaluation
  - `functions.py`: Implements Boolean function classes
  - `evolve.py`: Implements the evolutionary algorithm
  - `utils/`: Utility modules
    - `visualization.py`: Plotting functions
    - `io.py`: File I/O utilities
- `main.py`: Main script for running experiments

## Expected Outcomes

Based on theoretical predictions:

1. **Monotone Conjunctions/Disjunctions**: High success rate, polynomial number of generations
2. **Parity Functions**: Low success rate (not evolvable)
3. **Majority Functions**: Moderate success rate (more complex)

## Future Extensions (Phase 2)

- Testing with biased and clustered distributions
- Exploring the impact of different mutation operators
- Implementing crossover/recombination
- Testing with changing ("drifting") targets

## References

- Valiant, L. G. (2009). Evolvability. Journal of the ACM, 56(1), 1-21.
- Feldman, V. (2008). Evolvability from learning algorithms.
- Kanade, V. (2011). Evolution with recombination.
- Diochnos, D. I., & Turán, G. (2009). On evolvability: The swapping algorithm, product distributions, and covariance.

## License

MIT
