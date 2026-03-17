# CAS781_finalproject

This project implements and evaluates a Q-learning agent on the FrozenLake environment using Gymnasium. And apply Empirical method to analysis.

The goal is to:
- Understand reinforcement learning behavior in stochastic environments
- Evaluate performance under different hyperparameters
- Provide reproducible empirical results

## Research Questions

- How do learning rate (α), discount factor (γ), and exploration rate (ε) affect Q-learning performance?
- What hyperparameter combinations lead to stable convergence?
- How does stochasticity in FrozenLake impact learning?

## Methodology

We use a quantitative empirical approach:

- Environment: FrozenLake-v1 (stochastic setting)
- Algorithm: Tabular Q-learning
- Evaluation metrics:
  - Mean reward
  - Episode length
  - Convergence behavior

Experiments are repeated across multiple seeds.

## Study Design

We perform a controlled experiment:

- Independent variables:
  - α, γ, ε

- Dependent variables:
  - reward
  - episode length

- Control variables:
  - environment
  - episode count

## Project Structure

src/
  ├── env/
  │     └── env_factory.py
  ├── agents/
  │     └── q_learning.py
  ├── utils/
  │     ├── plotting.py
  │     └── logging.py
  ├── experiments/
  │     └── grid_search.py

results/
  ├── csv/
  └── plots/

README.md
requirements.txt

## How to Run

Install dependencies:

pip install -r requirements.txt

Run training:

python main.py

Run grid search:

python experiments/grid_search.py

    ## Results

Example output:

- Mean reward: 0.0
- Episode length: 100

Plots:
- Training reward curve
- Convergence behavior

## Reproducibility

- Random seeds are fixed
- Results are saved as CSV
- Plots are generated automatically

## Future Work

- Compare with SARSA
- Add function approximation
- Evaluate on larger environments