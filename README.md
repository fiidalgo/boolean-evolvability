# Boolean Evolvability Simulator

This project implements a computational model of Valiant's evolvability framework, as introduced in his 2009 paper "Evolvability". The simulator tests whether different classes of Boolean functions can evolve under the constraints of Darwinian evolution, viewed as a restricted form of PAC learning.

## Background

Leslie Valiant's 2009 paper "Evolvability" introduced a learning-theoretic model of Darwinian evolution, showing that certain classes of Boolean functions can "evolve" with polynomial resources under restricted conditions. In this model:

- Evolution is viewed as a constrained form of PAC learning
- Each generation produces slight random variants (mutations) of the current hypothesis
- Selection is guided only by aggregate performance on random samples
- There is no direct access to individual examples (statistical query model)

The model proves that monotone conjunctions and disjunctions are evolvable under the uniform distribution, while functions like parity are not evolvable in this framework.

## Implementation

This implementation adheres to the original constraints of Valiant's model and extends it with additional configurations:

1. **Example distributions**:

   - Uniform distribution (standard)
   - Binomial distribution
   - Beta distribution
   - Biased Bernoulli distribution (p=0.75)

2. **Empirical fitness evaluation**: Defines fitness as the fraction of examples on which the hypothesis agrees with the target concept

3. **Mutation configurations**:

   - Standard mutations (adding/removing one variable at a time)
   - No neutral mutations (only beneficial mutations accepted)
   - Smart initialization (starting with an intelligently pre-selected hypothesis)

4. **Selection based on fitness**: Selection follows statistical query learning principles

5. **Clear stopping criteria**: Evolution succeeds when accuracy reaches 1-ε

### Function Classes Tested

The simulator tests the following Boolean function classes:

- **Monotone Conjunctions**: AND of a subset of variables
- **Monotone Disjunctions**: OR of a subset of variables
- **General Conjunctions**: AND of a subset of variables or their negations
- **General Disjunctions**: OR of a subset of variables or their negations
- **Parity Functions**: XOR of a subset of variables
- **Majority Functions**: Returns 1 if majority of inputs are 1

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

### Running Comprehensive Experiments

To run the full suite of experiments with all configurations:

```bash
python run_experiments.py --n-values 5 10 20 30 50 --trials 5 --verbose
```

### Command Line Arguments

- `--function-classes`: List of function classes to test (conjunction, disjunction, general_conjunction, general_disjunction, parity, majority)
- `--n-values`: List of input sizes to test
- `--trials`: Number of trials for each configuration
- `--epsilon`: Target error threshold (default: 0.05)
- `--sample-size`: Number of examples for fitness evaluation (default: 1000)
- `--verbose`: Print detailed progress
- `--single-run`: Run a single experiment for testing

## Results

Results are saved in the `results/` directory with timestamps. Key findings from our experiments include:

### Standard Configuration

- MonotoneConjunction: 100% success rate, avg 1.0 generations
- MonotoneDisjunction: 100% success rate, avg 1.0 generations
- GeneralConjunction: 100% success rate, avg 1.0 generations
- GeneralDisjunction: 100% success rate, avg 1.0 generations
- Majority: 60% success rate, avg 39.0 generations
- Parity: 0% success rate, avg 465.6 generations (not evolvable)

### No Neutral Mutations Configuration

- MonotoneConjunction: 100% success rate, avg 1.0 generations
- MonotoneDisjunction: 100% success rate, avg 1.0 generations
- GeneralConjunction: 100% success rate, avg 1.0 generations
- GeneralDisjunction: 100% success rate, avg 1.0 generations
- Majority: 0% success rate, avg 72.6 generations
- Parity: 0% success rate, avg 64.6 generations

### Smart Initialization Configuration

- MonotoneConjunction: 100% success rate, avg 1.0 generations
- MonotoneDisjunction: 100% success rate, avg 1.0 generations
- GeneralConjunction: 100% success rate, avg 1.0 generations
- GeneralDisjunction: 100% success rate, avg 1.0 generations
- Majority: 40% success rate, avg 39.2 generations
- Parity: 0% success rate, avg 339.4 generations

### Distribution Effects

- Binomial distribution generally improved evolvability for Majority (100% success)
- Beta distribution showed mixed effects depending on function class
- Biased distribution (p=0.75) improved success rates across most functions

## Project Structure

- `evolvability/`: Main package
  - `environment.py`: Handles example generation and fitness evaluation
  - `functions.py`: Implements Boolean function classes
  - `evolve.py`: Implements the evolutionary algorithm
  - `utils/`: Utility modules
    - `visualization.py`: Plotting functions
    - `io.py`: File I/O utilities
- `main.py`: Script for running basic experiments
- `run_experiments.py`: Script for running comprehensive experiments
- `generate_report.py`: Script for generating experiment reports
- `results/`: Directory for experimental results
- `report/`: Directory containing visualizations and summaries

## Conclusions

Based on our empirical investigations:

1. **Theoretical Predictions Confirmed**: Monotone conjunctions and disjunctions are easily evolvable, while parity functions are not evolvable, confirming Valiant's theoretical predictions.

2. **Distribution Sensitivity**: Non-uniform distributions can significantly impact evolvability, sometimes making previously difficult functions more evolvable.

3. **Neutral Mutations Matter**: Allowing neutral mutations appears critical for complex functions like Majority, which cannot evolve without them.

4. **Initialization Effects**: Smart initialization can help in some cases but doesn't overcome fundamental limits of evolvability.

5. **Function Complexity**: General conjunctions and disjunctions are as evolvable as monotone ones, suggesting evolvability is more about function structure than complexity.

## References

- Valiant, L. G. (2009). Evolvability. Journal of the ACM, 56(1), 1-21.
- Feldman, V. (2008). Evolvability from learning algorithms.
- Kanade, V. (2011). Evolution with recombination.
- Diochnos, D. I., & Turán, G. (2009). On evolvability: The swapping algorithm, product distributions, and covariance.
