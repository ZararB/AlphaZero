# AlphaZero Chess Engine

An implementation of DeepMind's AlphaZero algorithm for chess. This project implements the core components: Monte Carlo Tree Search (MCTS) with neural network guidance, a deep residual network for position evaluation and move prediction, and a self-play training loop that generates its own training data.

## What is AlphaZero?

AlphaZero is a reinforcement learning algorithm that learns to play games through self-play. Unlike traditional chess engines that use hand-crafted evaluation functions, AlphaZero starts with only the game rules and learns strategy entirely through playing against itself. The algorithm combines Monte Carlo Tree Search for game tree exploration with a neural network that learns to evaluate positions and predict good moves.

## How It Works

The training process follows a simple but effective loop:

1. **Self-play**: The current network plays games against itself using MCTS to select moves
2. **Data collection**: Each game generates a sequence of positions with search statistics
3. **Training**: The network learns to predict game outcomes and match the move distributions from MCTS
4. **Iteration**: The improved network generates better training games, creating a self-improving cycle

The key insight is that MCTS provides high-quality training targets (both the final game outcome and the move probabilities from search), and the neural network learns to approximate this search in a single forward pass.

## Architecture

The system has three main components:

```
Self-Play (Multi-threaded)
    ↓
Replay Buffer
    ↓
Neural Network
```

**Self-Play**: Multiple threads run games in parallel. Each game uses MCTS to select moves, where the neural network provides position evaluations and move priors.

**Replay Buffer**: Stores completed games and samples positions for training. Positions are sampled proportionally to game length to balance training data.

**Neural Network**: A ResNet-style architecture that takes an 8×8×18 board representation and outputs:
- Value: Position evaluation from the current player's perspective (range [-1, 1])
- Policy: Probability distribution over all legal moves

## MCTS Implementation

The Monte Carlo Tree Search uses the standard four phases:

- **Selection**: Traverse the tree using UCB formula to balance exploration and exploitation
- **Expansion**: Create new nodes for legal moves, using network policy as priors
- **Evaluation**: Use the neural network to evaluate leaf positions
- **Backpropagation**: Propagate values up the tree, flipping perspective for the opponent

The UCB formula used is:

```
UCB(s,a) = Q(s,a) + c_puct × P(s,a) × (√N(s) / (1 + N(s,a)))
```

Where Q is the average action value, P is the prior from the network, N(s) is the parent visit count, and c_puct controls exploration.

## Neural Network

The network architecture follows the AlphaZero paper:

- **Input**: 8×8×18 feature planes encoding piece positions, castling rights, repetition count, and player to move
- **Backbone**: 7 residual blocks with 3×3 convolutions and skip connections
- **Value head**: Dense layers outputting a single scalar (tanh activation)
- **Policy head**: Dense layer with softmax over the full action space (1968 moves)

The action space includes all possible chess moves in UCI format: pawn moves, piece moves, castling, promotions, etc.

## Getting Started

### Installation

#### macOS Setup (Recommended)

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Linux/Ubuntu

```bash
pip install -r requirements.txt
```

Main dependencies:
- Python 3.9+ (macOS) or Python 3.7+ (Linux)
- TensorFlow 2.20.0+ (macOS) or TensorFlow 2.1.0+ (Linux)
- python-chess 1.999+
- NumPy

### Running Training

```bash
# macOS
./venv/bin/python __main__.py

# Or activate venv first
source venv/bin/activate
python __main__.py
```

This starts the training loop. The system will:
- Spawn multiple threads for parallel self-play
- Generate games and save them to the `games/` directory
- Train the network on batches sampled from the replay buffer
- Save model checkpoints to `models/` every 10 epochs

### Sample Training Output

Here's what a training session looks like:

```
======================================================================
AlphaZero Training Session - Sample Run
======================================================================

Configuration:
  Actors (parallel threads): 2
  Games per actor: 3
  MCTS simulations per move: 30
  Action space size: 1968
  Training batch size: 4

======================================================================
Epoch 1/3
======================================================================

[Self-Play Phase] Generating games...

  Actor 0, Game 1/3: Result=*, Moves=150, Value=0.00, Time=223.1s
  Actor 1, Game 1/3: Result=*, Moves=150, Value=0.00, Time=230.7s
  Actor 0, Game 2/3: Result=1-0, Moves=87, Value=1.00, Time=125.3s
  Actor 1, Game 2/3: Result=0-1, Moves=92, Value=-1.00, Time=131.2s

  Generated 6 games in 644.3s
  Total games in buffer: 6

[Training Phase] Updating network...

  ✓ Training step 1 completed in 2.34s
  Learning rate: 0.200000
  ✓ Checkpoint saved: models/checkpoint_0.h5 (25.1 MB)

  Epoch 1 completed in 646.6s

======================================================================
Training Session Complete!
======================================================================

Summary:
  Total epochs: 3
  Total games generated: 18
  Total training steps: 3
  Checkpoints saved: 3
```

**Key Metrics:**
- **Result**: Game outcome (`1-0` = white wins, `0-1` = black wins, `1/2-1/2` = draw, `*` = ongoing/max moves)
- **Moves**: Number of moves in the game
- **Value**: Terminal value from perspective of player to move (-1 = loss, 0 = draw, 1 = win)
- **Time**: Time taken to generate the game
- **Checkpoints**: Model weights saved periodically for resuming training

### Verifying Checkpoints

You can verify that checkpoints work correctly:

```bash
./venv/bin/python verify_checkpoints.py
```

This will:
- Load existing checkpoints or create a test one
- Verify inference works
- Test MCTS with loaded model
- Confirm training can continue from checkpoint

### Configuration

Training parameters are in `config.py`. Key settings:

```python
num_actors = 3              # Parallel self-play threads
num_games_per_actor = 5     # Games per thread per epoch
num_simulations = 50        # MCTS simulations per move
batch_size = 4              # Training batch size
training_steps = 700000     # Total training steps
```

You can adjust these based on your hardware. More simulations and actors will generate better training data but take longer.

## Project Structure

```
AlphaZero/
├── __main__.py          # Main training loop
├── config.py            # Configuration and move generation
├── game.py              # Chess game logic
├── mcts.py              # MCTS implementation
├── network.py           # Neural network architecture
├── node.py              # MCTS tree node
├── replayBuffer.py      # Experience replay buffer
├── sharedStorage.py     # Weight storage
├── client.py            # Distributed inference client
├── server.py            # Distributed inference server
└── resources/
    └── alphazero_preprint.pdf
```

## Distributed Inference

The code supports running inference on a remote server. This allows multiple training processes to share a single GPU for neural network evaluation:

```python
# Start server
python server.py

# In training code
network = Network(config, remote=True)
```

The client-server communication uses sockets with pickle for serialization.

## Implementation Notes

**State Representation**: The board is encoded as 18 feature planes. Twelve planes represent piece positions (6 piece types × 2 colors), four planes encode castling rights, one plane tracks repetition count, and one indicates the player to move.

**Move Generation**: All legal chess moves are pre-generated and stored in a lookup table. This includes pawn moves, piece moves, castling, promotions, and en passant. The action space has 1968 possible moves.

**Temperature Sampling**: Early moves use temperature sampling (proportional to visit counts) for exploration, while later moves use greedy selection (most visited child) for exploitation.

**Learning Rate Schedule**: The learning rate decays over training steps: starts at 0.2, drops to 0.02 at step 100k, then 0.002 at 300k, and 0.0002 at 500k.

## Training Tips

- Start with fewer simulations (25-50) to generate games faster during early training
- Monitor the replay buffer size - it's capped at 1M positions by default
- Checkpoint frequently - training can take a long time
- GPU acceleration helps significantly for network inference and training

## References

- [AlphaZero Paper](https://arxiv.org/abs/1712.01815) - "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"
- [python-chess](https://python-chess.readthedocs.io/) - Chess library used for game logic

## Recent Updates (macOS Build)

This project has been updated and fixed for macOS compatibility:

- ✅ **TensorFlow 2.x compatibility**: Updated from TF 1.x to TF 2.20 with eager execution
- ✅ **Critical bug fixes**: Policy normalization, value backpropagation, batch training
- ✅ **Complete move generation**: All chess pieces and move types supported
- ✅ **Checkpoint verification**: Models can be saved, loaded, and training resumed
- ✅ **Thread safety**: Proper locking for multi-threaded self-play

See `FIXES_APPLIED.md` for complete list of fixes and `MACOS_SETUP.md` for macOS-specific setup.

## Notes

This is a complete implementation of the AlphaZero algorithm. Training a strong chess engine requires significant computational resources and time. The original AlphaZero used thousands of TPUs and trained for days. This implementation can run on a single machine but will take much longer to reach competitive strength.

For production use, consider GPU acceleration, more parallel actors, and extended training periods. The architecture is designed to be game-agnostic and could be adapted for other games by modifying the game logic and action space.

## Troubleshooting

**macOS SSL Warning**: The urllib3 warning about OpenSSL/LibreSSL is harmless and can be ignored.

**Checkpoint Loading**: If you get errors loading checkpoints, ensure you're using the same TensorFlow version that created them.

**Training Speed**: Games can take 2-4 minutes each with 30 simulations. Reduce `num_simulations` for faster training during development.
