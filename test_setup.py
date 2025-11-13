"""Quick test to verify the setup works before training."""

from config import Config
from game import Game
from network import Network
from mcts import run_mcts
import numpy as np

print("Testing AlphaZero setup...")

# Test config
print("\n1. Testing Config...")
config = Config()
print(f"   ✓ Action space size: {config.num_actions}")
print(f"   ✓ Move list generated: {len(config.moveList)} moves")

# Test game
print("\n2. Testing Game...")
game = Game()
print(f"   ✓ Game initialized")
print(f"   ✓ Legal actions: {len(game.legal_actions())}")

# Test network
print("\n3. Testing Network...")
try:
    network = Network(config, remote=False)
    print(f"   ✓ Network initialized")
    
    # Test inference
    image = game.make_image()
    value, policy = network.inference(image)
    print(f"   ✓ Inference works: value shape={value.shape}, policy shape={policy.shape}")
except Exception as e:
    print(f"   ✗ Network error: {e}")
    raise

# Test MCTS
print("\n4. Testing MCTS...")
try:
    action, root = run_mcts(config, game, network)
    print(f"   ✓ MCTS completed: selected action={action}")
    print(f"   ✓ Root node has {len(root.children)} children")
except Exception as e:
    print(f"   ✗ MCTS error: {e}")
    raise

print("\n✅ All tests passed! Ready for training.")

