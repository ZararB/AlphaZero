# AlphaZero Chess Engine

A complete implementation of DeepMind's **AlphaZero** algorithm for chess, featuring Monte Carlo Tree Search (MCTS), deep residual neural networks, and self-play reinforcement learning. This project demonstrates advanced machine learning techniques including distributed computing, parallel game generation, and sophisticated search algorithms.

## 🎯 Overview

AlphaZero is a groundbreaking reinforcement learning algorithm that achieved superhuman performance in chess, shogi, and Go without any human knowledge beyond the game rules. This implementation recreates the core architecture from the [AlphaZero paper](https://arxiv.org/abs/1712.01815), featuring:

- **Monte Carlo Tree Search (MCTS)** with Upper Confidence Bound (UCB) selection
- **Deep Residual Neural Network** with dual outputs (value and policy)
- **Self-Play Training Loop** with parallel game generation
- **Experience Replay Buffer** with prioritized sampling
- **Distributed Inference Architecture** for scalable training

## 🏗️ Architecture

### Core Components

```
┌─────────────────┐
│   Self-Play     │  ← Parallel game generation with MCTS
│   (Multi-thread)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Replay Buffer  │  ← Experience storage & sampling
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Neural Network  │  ← ResNet architecture (value + policy)
└─────────────────┘
```

### Algorithm Flow

1. **Self-Play**: Multiple parallel actors play games using MCTS guided by the neural network
2. **Data Collection**: Game trajectories are stored in a replay buffer with search statistics
3. **Training**: The network learns to predict game outcomes (value) and optimal moves (policy)
4. **Iteration**: The improved network generates better self-play games, creating a self-improving loop

## 🔬 Technical Highlights

### Monte Carlo Tree Search (MCTS)

- **Selection**: UCB formula balances exploration vs exploitation
- **Expansion**: Neural network provides prior probabilities for legal moves
- **Evaluation**: Network predicts position value from current player's perspective
- **Backpropagation**: Values propagate up the tree with proper perspective flipping
- **Dirichlet Noise**: Root node exploration noise for diverse game play

### Neural Network Architecture

- **Input**: 8×8×18 feature planes (piece positions, castling rights, repetitions, player to move)
- **Backbone**: 7 residual blocks with skip connections (ResNet-style)
- **Outputs**:
  - **Value Head**: Position evaluation (tanh activation, range [-1, 1])
  - **Policy Head**: Move probability distribution (softmax over 1968 legal moves)

### Advanced Features

- **Parallel Self-Play**: Multi-threaded game generation for efficient data collection
- **Distributed Inference**: Client-server architecture for remote neural network evaluation
- **Prioritized Sampling**: Replay buffer samples positions proportional to game length
- **Temperature Sampling**: Exploration during early game moves, exploitation later
- **Learning Rate Scheduling**: Adaptive learning rate decay over training steps

## 📋 Requirements

- Python 3.7+
- TensorFlow 2.1.0
- python-chess 0.31.4
- NumPy 1.18.5

See `requirements.txt` for complete dependency list.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd AlphaZero

# Install dependencies
pip install -r requirements.txt
```

### Verify Setup

```bash
python test_setup.py
```

This verifies:
- Configuration loads correctly
- Game engine initializes
- Neural network builds successfully
- MCTS runs without errors

### Training

```bash
python __main__.py
```

The training process will:
1. Generate self-play games using MCTS (configurable number of parallel actors)
2. Store game trajectories in the replay buffer
3. Train the neural network on sampled positions
4. Save model checkpoints every 10 epochs to `models/` directory

### Configuration

Edit `config.py` to customize training parameters:

```python
# Parallelization
num_actors = 3                    # Number of parallel self-play threads
num_games_per_actor = 5           # Games per thread per epoch

# MCTS
num_simulations = 50              # MCTS simulations per move
num_sampling_moves = 30           # Temperature sampling for first N moves

# Training
batch_size = 4                    # Training batch size
training_steps = 700000           # Total training steps
learning_rate_schedule = {...}    # Adaptive learning rate schedule
```

## 📁 Project Structure

```
AlphaZero/
├── __main__.py          # Main training loop with parallel self-play
├── config.py            # Configuration and chess move generation
├── game.py              # Chess game logic and state representation
├── mcts.py              # Monte Carlo Tree Search implementation
├── network.py           # Neural network architecture (ResNet)
├── node.py              # MCTS tree node structure
├── replayBuffer.py      # Experience replay buffer with sampling
├── sharedStorage.py     # Network weight storage
├── client.py            # Distributed inference client
├── server.py            # Distributed inference server
└── resources/
    └── alphazero_preprint.pdf  # Original AlphaZero paper
```

## 🎓 Key Implementation Details

### Chess Move Representation

- **Action Space**: 1968 possible moves in UCI format (e.g., "e2e4", "e1g1")
- **Move Generation**: Complete move generation for all piece types:
  - Pawn moves (forward, capture, en passant, promotion)
  - Rook, Bishop, Queen moves (sliding pieces)
  - Knight moves (L-shaped)
  - King moves (including castling)

### State Representation

The neural network receives an 8×8×18 tensor encoding:
- **12 planes**: Piece positions (6 piece types × 2 colors)
- **4 planes**: Castling rights (white/black, kingside/queenside)
- **1 plane**: Repetition count
- **1 plane**: Player to move indicator

### MCTS UCB Formula

```
UCB(s,a) = Q(s,a) + c_puct × P(s,a) × (√N(s) / (1 + N(s,a)))

Where:
- Q(s,a): Average action value
- P(s,a): Prior probability from neural network
- N(s): Parent visit count
- N(s,a): Child visit count
- c_puct: Exploration constant
```

## 🔧 Advanced Usage

### Distributed Training

The implementation supports distributed inference where multiple clients can connect to a central server for neural network evaluation:

```python
# Server side
python server.py

# Client side (in config)
network = Network(config, remote=True)
```

### Custom Game Variants

The architecture is designed to be game-agnostic. To adapt for other games:
1. Modify `game.py` for game-specific logic
2. Update `config.py` for action space and move generation
3. Adjust neural network input dimensions in `network.py`

## 📊 Training Progress

Monitor training through:
- **Console Output**: Game results, training steps, model saves
- **Model Checkpoints**: Saved to `models/checkpoint_{epoch}.h5`
- **Game Replays**: Individual games saved to `games/` directory

## 🎯 Performance Considerations

- **MCTS Simulations**: More simulations = stronger play but slower training
- **Batch Size**: Larger batches improve training stability but require more memory
- **Parallel Actors**: More threads = faster data generation but higher CPU usage
- **Network Depth**: Deeper networks can learn more but train slower

## 📚 References

- [AlphaZero Paper](https://arxiv.org/abs/1712.01815) - "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"
- [python-chess Documentation](https://python-chess.readthedocs.io/) - Chess library used for game logic

## 🛠️ Technical Skills Demonstrated

This project showcases expertise in:

- **Reinforcement Learning**: Self-play, MCTS, value/policy learning
- **Deep Learning**: ResNet architectures, dual-head networks, batch training
- **Concurrent Programming**: Multi-threaded self-play, thread-safe data structures
- **Distributed Systems**: Client-server architecture for scalable inference
- **Algorithm Implementation**: Complex search algorithms, UCB selection
- **Software Engineering**: Modular design, configuration management, checkpointing

## 📝 License

This implementation is for educational and research purposes.

---

**Note**: This is a complete implementation of the AlphaZero algorithm. Training a competitive chess engine requires significant computational resources and time. For production use, consider GPU acceleration and extended training periods.
