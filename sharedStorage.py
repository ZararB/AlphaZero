from network import Network


class SharedStorage(object):

	def __init__(self, config):
		self._networks = []
		self.config = config

	def load_network(self):

		if self._networks:
			return self._networks[-1]
		else:
			return Network(self.config)
			
	def latest_network(self):
		if not self._networks:
			network = Network(self.config)
			self._networks.append(network)
	  
		return self._networks[-1] 

	def save_network(self, network):
		if len(self._networks) > 2:
			self._networks = self._networks[1:]

		self._networks.append(network)