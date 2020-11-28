from network import Network


class SharedStorage(object):

	def __init__(self, network):
		self.weights = [network.model.get_weights()]


	def load_weights(self):
		return self.weights[-1]

			
	def save_weights(self, network):

		if len(self.weights) > 2:
			self.weights = self.weights[1:]

		self.weights.append(network.model.get_weights())