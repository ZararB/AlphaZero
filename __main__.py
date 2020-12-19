from game import Game
from config import Config
from node import Node 
from replayBuffer import ReplayBuffer
from network import Network
import math
import numpy as np
import random
import tensorflow as tf
import threading
import os 
import time
import pickle as pkl
from mcts import run_mcts
from tensorflow.keras.models import load_model





def play_game(config: Config, network: Network):

	game = Game()

	while not game.terminal() and len(game.history) < config.max_moves:

		action, root = run_mcts(config, game, network)
		game.apply(action)
		game.store_search_statistics(root)

	return game 


class SelfPlay(threading.Thread):
	
	def run(self):

		for i in range(config.num_games_per_actor):
			
			game = play_game(config, network)
			result = game.board.result()
			terminal_value = game.terminal_value()


			if result == '1/2-1/2':
				file_name = 'games/draw_' + str(np.random.randint(500000))
			else:
				file_name = 'games/' + result + '_' + str(np.random.randint(500000))

			with open(file_name, 'wb') as f:
				pkl.dump(game, f)
			
			replay_buffer.save_game(game)
			print('Thread: {}, Game: {}, Result {}, Reward {}'.format(threading.get_ident(), i, result, terminal_value))
			
			


if __name__ == '__main__':


	config = Config()
	replay_buffer = ReplayBuffer(config)

	model_files = os.listdir('models/')
	model_files.sort()

	if model_files:
		model_filepath = 'models/' + model_files[-1]
		network = Network(config, model_filepath)
	else:
		network = Network(config)

	num_epochs = 1000000

	for e in range(num_epochs):
		
		# Make network read-only so it can be run on multiple threads
		network.graph.finalize()

		jobs = []

		for _ in range(config.num_actors):
			job = SelfPlay()
			job.start()
			jobs.append(job)

		# Wait for all threads to complete 
		for job in jobs:
			job.join()

		# Make network writable for parameter update
		network.graph._unsafe_unfinalize()
		 
		batch = replay_buffer.sample_batch()
		network.update(batch)
			
		if e % 10 == 0 :
			network.model.save('models/{}.h5'.format(e))