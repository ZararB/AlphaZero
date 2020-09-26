from game import Game
from config import Config
from sharedStorage import SharedStorage
from node import Node 
from replayBuffer import ReplayBuffer

def alphazero(config: Config):

	storage = SharedStorage()
	replay_buffer = ReplayBuffer(config)

	
    game = run_selfplay(config, storage, replay_buffer)

  train_network(config, storage, replay_buffer)

  return storage.latest_network()



##################################
####### Part 1: Self-Play ########



def run_selfplay(config: AlphaZeroConfig, storage: SharedStorage,
                 replay_buffer: ReplayBuffer):
  while True:
    network = storage.latest_network()
    game = play_game(config, network)
    replay_buffer.save_game(game)