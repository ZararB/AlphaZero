class Config(object):
		
	def __init__(self):

		self.num_actors = 2
		self.num_games_per_actor = 1
		self.num_sampling_moves = 30
		self.max_moves = 512  # for chess and shogi, 722 for Go.
		self.num_simulations = 10
		# Root prior exploration noise.
		self.root_dirichlet_alpha = 0.3  # for chess, 0.03 for Go and 0.15 for shogi.
		self.root_exploration_fraction = 0.25
		# UCB formula
		self.pb_c_base = 19652
		self.pb_c_init = 1.25
				### Training
		self.training_steps = int(700e3)
		self.checkpoint_interval = int(1e3)
		self.window_size = int(1e3)
		self.batch_size = 4098
		self.weight_decay = 1e-4
		self.momentum = 0.9
		# Schedule for chess and shogi, Go starts at 2e-2 immediately.
		self.learning_rate_schedule = {
				0: 2e-1,
				100e3: 2e-2,
				300e3: 2e-3,
				500e3: 2e-4
		}
		self.moveList = self.generateMoveList()
		self.num_actions = len(self.moveList)

	def generateMoveList(self):

		moves = self.generateQueenMoves() + self.generateKnightMoves() + self.generatePawnPromotions()
		
		return moves
		 
	def generateQueenMoves(self):
		moves = []
	
		files = ['a','b','c','d','e','f','g','h']
		ranks = [str(x) for x in range(1,9)]

		rows = list(range(1,9))
		cols = list(range(1,9))

		for col in cols:
			for row in rows:

				currentSquare = files[col-1] + ranks[row-1]

				for direction in ['n','ne','e','se','s','sw','w','nw']:
					for numSquares in range(1,8):

						if direction == 'n':
							newRow = row + numSquares
							newCol = col 
						elif direction == 'ne':
							newRow = row + numSquares
							newCol = col + numSquares
						elif direction == 'e':
							newRow = row 
							newCol = col + numSquares
						elif direction == 'se':
							newRow = row - numSquares
							newCol = col + numSquares
						elif direction == 's':
							newRow = row - numSquares
							newCol = col
						elif direction == 'sw':
							newRow = row - numSquares
							newCol = col - numSquares
						elif direction == 'w':
							newRow = row 
							newCol = col - numSquares
						elif direction == 'nw':
							newRow = row + numSquares
							newCol = col - numSquares

						if self.isSquareValid(newCol, newRow):
							newSquare = files[newCol-1] + ranks[newRow-1]
							move = currentSquare + newSquare
							moves.append(move)

		return moves 

	def generateKnightMoves(self):
		moves = []

		files = ['a','b','c','d','e','f','g','h']
		ranks = [str(x) for x in range(1,9)]

		rows = list(range(1,9))
		cols = list(range(1,9))

		for col in cols:
			for row in rows:
				currentSquare = files[col-1] + ranks[row-1]

				for direction in ['ul', 'ur', 'ru', 'rd', 'dr','dl', 'ld','lu']:

					if direction == 'ul':
						newCol = col - 1 
						newRow = row + 2 
					elif direction == 'ur':
						newCol = col + 1 
						newRow = row + 2
					elif direction == 'ru':
						newCol = col + 2
						newRow = row + 1
					elif direction == 'rd':
						newCol = col + 2 
						newRow = row - 1 
					elif direction == 'dr':
						newCol = col + 1
						newRow = row - 2
					elif direction == 'dl':
						newCol = col - 1  
						newRow = row - 2
					elif direction == 'ld':
						newCol = col - 2 
						newRow = row - 1
					elif direction == 'lu':
						newCol = col - 2  
						newRow = row + 1

					if self.isSquareValid(newCol, newRow):
						newSquare = files[newCol-1] + ranks[newRow-1]
						move = currentSquare + newSquare
						moves.append(move)

		return moves

	def generatePawnPromotions(self):
		moves = []

		files = ['a','b','c','d','e','f','g','h']
		ranks = [str(x) for x in range(1,9)]

		rows = [2, 7]
		cols = list(range(1,9))

		for col in cols:
			for row in rows:

				currentSquare = files[col-1] + ranks[row-1]

				if row == 7: 

					for direction in ['nw', 'n', 'ne']:
						for piece in ['n', 'b', 'r', 'q']:

							if direction == 'nw':
								newRow = row + 1 
								newCol = col - 1 
							elif direction == 'n':
								newRow = row + 1 
								newCol = col 
							elif direction == 'ne':
								newRow = row + 1 
								newCol = col + 1 

							if self.isSquareValid(newCol, newRow):
								newSquare = files[newCol-1] + ranks[newRow-1]				
								move = currentSquare + newSquare + piece					
								moves.append(move)


				elif row == 2 :

					for direction in ['se', 's', 'sw']:
						for piece in ['n', 'b', 'r', 'q']:

							if direction == 'se':
								newRow = row - 1 
								newCol = col + 1 

							elif direction == 's':
								newRow = row - 1 
								newCol = col 

							elif direction == 'sw':
								newRow = row - 1
								newCol = col - 1 

							if self.isSquareValid(newCol, newRow):
								newSquare = files[newCol-1] + ranks[newRow-1]
								move = currentSquare + newSquare + piece
								moves.append(move)

		return moves
	
	def isSquareValid(self, col, row):

		if (row >=1 and row <= 8) and (col >= 1 and col <=8):
			return True
		else:
			return False
	