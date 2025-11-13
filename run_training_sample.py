"""Run a short training session to generate sample results for README."""

from game import Game
from config import Config
from replayBuffer import ReplayBuffer
from network import Network
import numpy as np
import threading
import os 
import pickle as pkl
from mcts import run_mcts
import time

def play_game(config: Config, network: Network):
	game = Game()
	while not game.terminal() and len(game.history) < config.max_moves:
		action, root = run_mcts(config, game, network)
		game.apply(action)
		game.store_search_statistics(root)
	return game 

class SelfPlay(threading.Thread):
	def __init__(self, config, network, replay_buffer, lock, actor_id):
		threading.Thread.__init__(self)
		self.config = config
		self.network = network
		self.replay_buffer = replay_buffer
		self.lock = lock
		self.actor_id = actor_id
	
	def run(self):
		for i in range(self.config.num_games_per_actor):
			start_time = time.time()
			game = play_game(self.config, self.network)
			game_time = time.time() - start_time
			
			result = game.board.result() 
			terminal_value = game.terminal_value()
			num_moves = len(game.history)
			
			with self.lock:
				self.replay_buffer.save_game(game)
			
			print(f'  Actor {self.actor_id}, Game {i+1}/{self.config.num_games_per_actor}: '
			      f'Result={result}, Moves={num_moves}, Value={terminal_value:.2f}, Time={game_time:.1f}s')

if __name__ == '__main__':
	print("=" * 70)
	print("AlphaZero Training Session - Sample Run")
	print("=" * 70)
	
	config = Config()
	# Use reasonable values for demonstration
	config.num_actors = 1  # Single actor for faster demo
	config.num_games_per_actor = 2  # Just 2 games
	config.num_simulations = 20  # Reduced for faster demo
	config.batch_size = 2
	config.max_moves = 50  # Shorter games for demo
	
	replay_buffer = ReplayBuffer(config)
	network = Network(config, remote=False)
	
	os.makedirs('models', exist_ok=True)
	os.makedirs('games', exist_ok=True)
	
	lock = threading.Lock()
	training_step = 0
	
	print(f'\nConfiguration:')
	print(f'  Actors (parallel threads): {config.num_actors}')
	print(f'  Games per actor: {config.num_games_per_actor}')
	print(f'  MCTS simulations per move: {config.num_simulations}')
	print(f'  Action space size: {config.num_actions}')
	print(f'  Training batch size: {config.batch_size}')
	print(f'  Max moves per game: {config.max_moves}')
	print(f'  Learning rate schedule: {config.learning_rate_schedule}\n')
	
	num_epochs = 2  # Just 2 epochs for demo
	
	for e in range(num_epochs):
		print(f"{'='*70}")
		print(f"Epoch {e+1}/{num_epochs}")
		print(f"{'='*70}")
		
		epoch_start = time.time()
		print('\n[Self-Play Phase] Generating games...\n')
		
		jobs = []
		for actor_id in range(config.num_actors):
			job = SelfPlay(config, network, replay_buffer, lock, actor_id)
			job.start()
			jobs.append(job)
		
		for job in jobs:
			job.join()
		
		selfplay_time = time.time() - epoch_start
		print(f'\n  Generated {config.num_actors * config.num_games_per_actor} games in {selfplay_time:.1f}s')
		print(f'  Total games in buffer: {len(replay_buffer.buffer)}')
		
		print('\n[Training Phase] Updating network...\n')
		train_start = time.time()
		
		batch = replay_buffer.sample_batch()
		if len(batch) > 0:
			network.update(batch, training_step)
			training_step += 1
			train_time = time.time() - train_start
			print(f'  ✓ Training step {training_step} completed in {train_time:.2f}s')
			print(f'  Learning rate: {network.get_learning_rate(training_step):.6f}')
		else:
			print('  ⚠ No samples in buffer yet')
		
		if e % 1 == 0:
			try:
				checkpoint_path = f'models/sample_checkpoint_{e}.h5'
				network.model.save(checkpoint_path)
				file_size = os.path.getsize(checkpoint_path) / (1024*1024)  # MB
				print(f'  ✓ Checkpoint saved: {checkpoint_path} ({file_size:.1f} MB)')
			except Exception as ex:
				print(f'  ✗ Error saving model: {ex}')
		
		epoch_time = time.time() - epoch_start
		print(f'\n  Epoch {e+1} completed in {epoch_time:.1f}s\n')
	
	print("=" * 70)
	print("Training Session Complete!")
	print("=" * 70)
	print(f"\nSummary:")
	print(f"  Total epochs: {num_epochs}")
	print(f"  Total games generated: {len(replay_buffer.buffer)}")
	print(f"  Total training steps: {training_step}")
	print(f"  Checkpoints saved: {len([f for f in os.listdir('models/') if 'sample_checkpoint' in f])}")
	print(f"\nCheckpoints available in: models/")
	print(f"Games saved in: games/")
	print("\nTo continue training, run: ./venv/bin/python __main__.py")

