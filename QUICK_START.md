# Quick Start Guide

## Installation

```bash
# Install required packages
pip install python-chess numpy tensorflow

# Or if using requirements.txt
pip install -r requirements.txt
```

## Test the Setup

```bash
python test_setup.py
```

This will verify:
- Config loads correctly
- Game can be created
- Network initializes
- MCTS runs without errors

## Run Training

```bash
python __main__.py
```

The training will:
1. Generate self-play games using MCTS
2. Save games to `games/` directory
3. Train the network on batches from replay buffer
4. Save model checkpoints to `models/` every 10 epochs

## Monitor Progress

Watch the console output for:
- Game results from each thread
- Model save confirmations
- Training step counts

## Adjust Settings

Edit `config.py` to change:
- `num_actors`: Number of parallel self-play threads
- `num_games_per_actor`: Games per thread per epoch
- `num_simulations`: MCTS simulations per move (more = better but slower)
- `batch_size`: Training batch size
- `num_sampling_moves`: Temperature sampling for exploration

## Troubleshooting

**Error: "ModuleNotFoundError: No module named 'chess'"**
- Install: `pip install python-chess`

**Error: "No module named 'tensorflow'"**
- Install: `pip install tensorflow`

**Error: "Move not in moveList"**
- This shouldn't happen with the complete move generation, but if it does, the code will skip the move gracefully

**Training is slow:**
- Reduce `num_simulations` for faster games
- Reduce `num_actors` and `num_games_per_actor` for testing
- Increase `batch_size` for more efficient training

