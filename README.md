# Multi-Objective Metro Network Expansion (`ours`)

A codebase for multi-objective metro network expansion planning using graph neural networks and reinforcement learning.

## 1. Directory Structure

- `data/`: Urban networks, existing lines, OD matrices, and static data files for multiple objectives
- `metro/`: Underlying physical environment for metro expansion, configurations, and basic network modules
- `khrylib/`: Core reinforcement learning utilities
- `ours/`: Core implementations of multi-objective potential encoding, policy networks, and PPO training
- `train.py`: Main entry point for model training and test evaluation
- `requirements.txt`: Python dependencies

## 2. Dependencies

Python 3.10+ and PyTorch with CUDA support are recommended. Install the dependencies with:

```bash
pip install -r requirements.txt
```

## 3. Training

Run the following command from the project root, explicitly providing three nonnegative preference coefficients:

```bash
python -B train.py --weights <w_G> <w_J> <w_F> --output <training_output_directory>
```

The program normalizes the coefficients into a preference vector that sums to one, then initializes the model from scratch, trains it, selects a model through validation, and performs testing. All run artifacts are written to the specified empty directory, which must be outside the source directory.

Higher values are better for all three metrics:

- G: Gain in segment-based OD benefits relative to the initial network.
- J: Gain in home-to-work commuting benefits relative to the initial network.
- F: Percentage-point improvement in station access accessibility, calculated using community weights and exponential distance decay.

The scalar training reward is the inner product of the three anchor-normalized marginal gains and the current training preference vector. Overall utility is calculated by dividing the terminal G/J/F values by their respective reference scales and summing them with the target preference weights. Construction cost is a hard budget constraint and is reported separately; it is not included as a negative reward term.

The training mechanism is selected automatically based on the largest component of the supplied preference vector: when the home-to-work commuting objective dominates, training uses a demand-fused graph, a potential prior, and a power-law preference curriculum; when fairness dominates, it uses the original demand graph and a sparse branching mechanism in the final stage; when ridership dominates or the largest components are tied, it uses channel-specific potentials and a sparse branching mechanism in the final stage. The curriculum affects training only; validation and testing always use the target preferences specified on the command line. Preferences and decoding modes are stored independently for each state.

Run parameters can be adjusted using `--iterations`, `--rollout-steps`, `--mini-batch-size`, `--num-optim-epoch`, `--seed`, and `--device`. No training results are included in the initial project directory.

## 4. Test Evaluation

```bash
python -B train.py --mode evaluate --weights <w_G> <w_J> <w_F> --checkpoint <model_file> --output <evaluation_output_directory>
```

Evaluation must use the same data, environment parameters, configuration, and target preferences as the model. Validation uses a separate, fixed sampling batch, while final testing uses three additional batches of random numbers; each batch includes both deterministic and stochastic decoding. The final plan is exported after independent verification of network shortest paths and asset costs.

## 5. Correctness Checks

```bash
python -B train.py --check --weights <w_G> <w_J> <w_F>
```

This command does not train the model or create output files. It checks zero gains in the initial state, valid stopping behavior, cumulative rewards, independent evaluation, marginal gains in service coverage areas, PPO probability consistency, and finite gradients for degenerate candidate sets. Static objective assets are strictly aligned with the current candidate network; the program will not apply them to a different set of nodes.
