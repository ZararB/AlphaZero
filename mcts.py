from node import Node 
import numpy as np
import math

def run_mcts(config, game, network):

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

def evaluate(node, game, network):
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

def select_child(config, node):

	_, action, child = max((ucb_score(config, node, child), action, child) 
							for action, child in node.children.items())

	return action, child

def ucb_score(config, parent, child):
	pb_c = math.log((parent.visit_count + config.pb_c_base + 1 ) / 
					config.pb_c_base) + config.pb_c_init

	pb_c *= math.sqrt(parent.visit_count) / (child.visit_count+1)

	prior_score = pb_c * child.prior 
	value_score = child.value()
	return prior_score + value_score



def add_exploration_noise(config, node):
	actions = node.children.keys()
	noise = np.random.gamma(config.root_dirichlet_alpha, 1, len(actions))
	frac = config.root_exploration_fraction

	for a, n in zip(actions, noise):
		node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac
