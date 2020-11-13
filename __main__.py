from game import Game
from config import Config
from sharedStorage import SharedStorage
from node import Node 
from replayBuffer import ReplayBuffer
from network import Network
import math
import numpy
import random
import tensorflow as tf
import threading
import time
from tensorflow.keras.models import load_model


##################################
####### Part 1: Self-Play ########


def run_selfplay(config: Config, storage: SharedStorage,
				 replay_buffer: ReplayBuffer):

	for i in range(config.num_games_per_epoch):
		network = storage.latest_network()
		game = play_game(config, network)
		replay_buffer.save_game(game)

def play_game(config: Config, network: Network):

	game = Game()

	while not game.terminal() and len(game.history) < config.max_moves:

		action, root = run_mcts(config, game, network)
		game.apply(action)
		game.store_search_statistics(root) #TODO figure out what this is doing

	return game 

def run_mcts(config: Config, game : Game, network : Network):

	root = Node(0)
	evaluate(root, game, network)
	#add_exploration_noise(config, root)

	for _ in range(config.num_simulations):
		node = root
		scratch_game = game.clone()
		search_path = [node]

		while node.expanded():
			action, node = select_child(config, node)
			scratch_game.apply(action)
			search_path.append(node)

		value = evaluate(node, scratch_game, network)
		backpropagate(search_path, value, scratch_game.to_play())

	return (select_action(config, game, root), root)

def evaluate(node: Node, game: Game, network: Network):
	value, policy = network.inference(game.make_image())
  
	# Expand the node.
	node.to_play = game.to_play()
	legal_actions = game.legal_actions()
	'''
	policy = {a: math.exp(policy[0][a]) for a in legal_actions}
	policy_sum = sum(policy.values())
	'''
	policy = {a: policy[0][a] for a in legal_actions}
	for action, p in policy.items():
		node.children[action] = Node(p)
	
	return value

# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(search_path, value, to_play):

	for node in search_path:
		node.value_sum += value if node.to_play == to_play else (1 - value)
		node.visit_count += 1

def select_action(config, game, root):

	visit_counts = [(child.visit_count, action) for action, child in root.children.items()]
	'''
	if (len(game.history) < config.num_sampling_moves):
		_, action = random.sample(visit_counts, 1)[0] # 
	else:
		_, action = max(visit_counts)
	'''
	_, action = max(visit_counts)

	return action 

def select_child(config : Config, node: Node):

	_, action, child = max((ucb_score(config, node, child), action, child) 
							for action, child in node.children.items())

	return action, child

def ucb_score(config : Config, parent: Node, child: Node):
	pb_c = math.log((parent.visit_count + config.pb_c_base + 1 ) / 
					config.pb_c_base) + config.pb_c_init

	pb_c *= math.sqrt(parent.visit_count) / (child.visit_count+1)

	prior_score = pb_c * child.prior 
	value_score = child.value()
	return prior_score + value_score



def add_exploration_noise(config: Config, node: Node):
	actions = node.children.keys()
	noise = numpy.random.gamma(config.root_dirichlet_alpha, 1, len(actions))
	frac = config.root_exploration_fraction

	for a, n in zip(actions, noise):
		node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac




class SelfPlay(threading.Thread):
	
	def run(self):

		for _ in range(config.num_games_per_actor):
			network1 = Network(config)
			network1.model = load_model('test.h5')
			game = play_game(config, network1)
			replay_buffer.save_game(game)
		
		return 0





######## Debugging ############


if __name__ == '__main__':

	config = Config()
	network = Network(config)
	network.model.save('test.h5')
	storage = SharedStorage(config)
	replay_buffer = ReplayBuffer(config)
	num_epochs = 10000

	time0 = time.time()
	for e in range(1):

		jobs = []

		for _ in range(config.num_actors):
			job = SelfPlay()
			job.start()
			jobs.append(job)

		for job in jobs:
			job.join()

		time1 = time.time()
		delta = time1 - time0
		#batch = replay_buffer.sample_batch()
		#network.update(batch)

		
	
	time0 = time.time()
	for e in range(2):
		game = play_game(config, network)
		replay_buffer.save_game(game)

	time1 = time.time()

	delta1 = time1 - time0
	print(delta1, delta)
