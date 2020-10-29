from network import Network

class SharedStorage(object):

  def __init__(self):
    self._networks = {}

    def load_network(self):

    	if self._networks:
    		return self._networks
    	else:
    		return initialize_network() # policy -> uniform, value -> 0.5


  def latest_network(self) -> Network:
    if not self._networks:
    	network = Network()
    	self._networks.append(network)
      
    return self._networks[-1] 

  def save_network(self, step: int, network: Network):
    self._networks[-1] = network