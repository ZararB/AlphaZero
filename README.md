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

```bash
pip install -r requirements.txt
```

Main dependencies:
- Python 3.7+
- TensorFlow 2.1.0
- python-chess 0.31.4
- NumPy

### Running Training

```bash
python __main__.py
```

This starts the training loop. The system will:
- Spawn multiple threads for parallel self-play
- Generate games and save them to the `games/` directory
- Train the network on batches sampled from the replay buffer
- Save model checkpoints to `models/` every 10 epochs

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

## Notes

This is a complete implementation of the AlphaZero algorithm. Training a strong chess engine requires significant computational resources and time. The original AlphaZero used thousands of TPUs and trained for days. This implementation can run on a single machine but will take much longer to reach competitive strength.

For production use, consider GPU acceleration, more parallel actors, and extended training periods. The architecture is designed to be game-agnostic and could be adapted for other games by modifying the game logic and action space.
