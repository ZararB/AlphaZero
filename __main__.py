from game import Game
from config import Config
from sharedStorage import SharedStorage
from node import Node 
from replayBuffer import ReplayBuffer
from network import Network
import math
import numpy
import random
from typing import List
import tensorflow as tf



##################################
####### Part 1: Self-Play ########


def select_action(config, game, root):

	visit_counts = [(child.visit_count, action) for action, child in root.children.iteritems()]
	if (len(game.history) < config.num_sampling_moves):
		_, action = random.sample(visit_counts)[0]
	else:
		_, action = max(visit_counts)

	return action 

def select_child(config : Config, node: Node):

	_, action, child = max((ucb_score(config, node, child), action, child) 
							for action, child in node.children.iteritems())

	return action, child

def ucb_score(config : Config, parent: Node, child: Node):
	pb_c = math.log((parent.visit_count + config.pb_c_base + 1 ) / 
					config.pb_c_base) + config.pb_c_init

	pb_c *= math.sqrt(parent.visit_count) / (child.visit_count+1)

	prior_score = pb_c * child.prior 
	value_score = child.value()
	return prior_score + value_score

# We use the neural network to obtain a value and policy prediction.
def evaluate(node: Node, game: Game, network: Network):
	value, policy_logits = network.inference(game.make_image(-1))
  
	# Expand the node.
	node.to_play = game.to_play()
	policy = {a: math.exp(policy_logits[a]) for a in game.legal_actions()}
	policy_sum = sum(policy.itervalues())
	for action, p in policy.iteritems():
		node.children[action] = Node(p / policy_sum)
	
	return value


# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(search_path: List[Node], value: float, to_play):

	for node in search_path:
		node.value_sum += value if node.to_play == to_play else (1 - value)
		node.visit_count += 1

def add_exploration_noise(config: Config, node: Node):
	actions = node.children.keys()
	noise = numpy.random.gamma(config.root_dirichlet_alpha, 1, len(actions))
	frac = config.root_exploration_fraction

	for a, n in zip(actions, noise):
		node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac



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
		game.store_search_statistics(root)

	return game 

def run_mcts(config: Config, game : Game, network : Network):

	root = Node(0)
	evaluate(root, game, network)
	add_exploration_noise(config, root)

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


config = Config()
network = Network(config)
play_game(config, network)

""" 


######### End Self-Play ##########
##################################

##################################
####### Part 2: Training #########


def train_network(config: Config, storage: SharedStorage,
				  replay_buffer: ReplayBuffer):

	network = storage.latest_network()
	optimizer = tf.train.MomentumOptimizer(config.learning_rate_schedule,
										   config.momentum)
										   
	for i in range(config.training_steps):
		if i % config.checkpoint_interval == 0:
			storage.save_network(i, network)
		batch = replay_buffer.sample_batch()
		update_weights(optimizer, network, batch, config.weight_decay)
	
	storage.save_network(config.training_steps, network)



def update_weights(optimizer: tf.train.Optimizer, network: Network, batch,
				   weight_decay: float):

	loss = 0
	for image, (target_value, target_policy) in batch:
		value, policy_logits = network.inference(image)
		loss += (
			tf.losses.mean_squared_error(value, target_value) +
			tf.nn.softmax_cross_entropy_with_logits(
				logits=policy_logits, labels=target_policy))
  
	for weights in network.get_weights():
		loss += weight_decay * tf.nn.l2_loss(weights)
  
	optimizer.minimize(loss)

if __name__ == '__main__':

	config = Config()
	storage = SharedStorage()
	replay_buffer = ReplayBuffer(config)
	num_epochs = 1
	
	for _ in range(num_epochs):
		run_selfplay(config, storage, replay_buffer)
		train_network(config, storage, replay_buffer)
		
"""
