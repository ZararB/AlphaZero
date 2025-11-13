"""Quick training test - runs just 2 epochs to verify everything works."""

from game import Game
from config import Config
from node import Node 
from replayBuffer import ReplayBuffer
from network import Network
import numpy as np
import threading
import os 
import pickle as pkl
from mcts import run_mcts

def play_game(config: Config, network: Network):
	game = Game()
	while not game.terminal() and len(game.history) < config.max_moves:
		action, root = run_mcts(config, game, network)
		game.apply(action)
		game.store_search_statistics(root)
	return game 

class SelfPlay(threading.Thread):
	def __init__(self, config, network, replay_buffer, lock):
		threading.Thread.__init__(self)
		self.config = config
		self.network = network
		self.replay_buffer = replay_buffer
		self.lock = lock
	
	def run(self):
		for i in range(self.config.num_games_per_actor):
			game = play_game(self.config, self.network)
			result = game.board.result() 
			terminal_value = game.terminal_value()
			
			with self.lock:
				self.replay_buffer.save_game(game)
			
			print(f'  Thread {threading.get_ident()}, Game {i+1}/{self.config.num_games_per_actor}, Result: {result}, Value: {terminal_value:.2f}')

if __name__ == '__main__':
	print("=" * 60)
	print("AlphaZero Training Test - 2 Epochs")
	print("=" * 60)
	
	config = Config()
	# Use smaller values for quick test
	config.num_actors = 2
	config.num_games_per_actor = 2
	config.num_simulations = 25  # Reduced for speed
	config.batch_size = 2
	
	replay_buffer = ReplayBuffer(config)
	network = Network(config, remote=False)
	
	os.makedirs('models', exist_ok=True)
	os.makedirs('games', exist_ok=True)
	
	lock = threading.Lock()
	training_step = 0
	
	print(f'\nConfiguration:')
	print(f'  Actors: {config.num_actors}')
	print(f'  Games per actor: {config.num_games_per_actor}')
	print(f'  Simulations: {config.num_simulations}')
	print(f'  Action space: {config.num_actions}')
	print(f'  Batch size: {config.batch_size}\n')
	
	for e in range(2):
		print(f'Epoch {e+1}/2:')
		print('  Generating self-play games...')
		
		jobs = []
		for _ in range(config.num_actors):
			job = SelfPlay(config, network, replay_buffer, lock)
			job.start()
			jobs.append(job)
		
		for job in jobs:
			job.join()
		
		print(f'  Training on batch...')
		batch = replay_buffer.sample_batch()
		if len(batch) > 0:
			network.update(batch, training_step)
			training_step += 1
			print(f'  ✓ Training step {training_step} completed')
		else:
			print('  ⚠ No samples in buffer yet')
		
		if e % 1 == 0:
			try:
				network.model.save(f'models/test_checkpoint_{e}.h5')
				print(f'  ✓ Model saved: models/test_checkpoint_{e}.h5')
			except Exception as ex:
				print(f'  ✗ Error saving model: {ex}')
		
		print()
	
	print("=" * 60)
	print("✅ Training test completed successfully!")
	print("=" * 60)
	print(f"\nTotal games generated: {len(replay_buffer.buffer)}")
	print(f"Total training steps: {training_step}")

