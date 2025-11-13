class Config(object):
		
	def __init__(self):
		
		self.HEADERSIZE = 10
		self.INFERENCE_FLAG = 2
		self.UPDATE_FLAG = 1 
		self.FLAGSIZE = 5
		self.num_actors = 3
		self.num_games_per_actor = 5
		self.num_sampling_moves = 30  # Temperature sampling for first N moves
		self.max_moves = 150  # for chess and shogi, 722 for Go.
		self.num_simulations = 50
		# Root prior exploration noise.
		self.root_dirichlet_alpha = 0.3  # for chess, 0.03 for Go and 0.15 for shogi.
		self.root_exploration_fraction = 0.25
		# UCB formula
		self.pb_c_base = 19652
		self.pb_c_init = 1.25
		### Training
		self.training_steps = int(700e3)
		self.checkpoint_interval = int(1e3)
		self.window_size = int(1e6)
		self.batch_size = 4
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
		"""Generate all possible chess moves in UCI format."""
		moves = (self.generatePawnMoves() + 
		         self.generateRookMoves() + 
		         self.generateBishopMoves() + 
		         self.generateQueenMoves() + 
		         self.generateKnightMoves() + 
		         self.generateKingMoves() + 
		         self.generatePawnPromotions() +
		         self.generateCastlingMoves())
		
		# Remove duplicates and sort for consistency
		moves = sorted(list(set(moves)))
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

	def generatePawnMoves(self):
		"""Generate all non-promotion pawn moves (forward, capture, en passant)."""
		moves = []
		files = ['a','b','c','d','e','f','g','h']
		ranks = [str(x) for x in range(1,9)]
		
		# White pawns (move up, ranks 2-7)
		for col in range(1, 9):
			for start_rank in range(2, 8):
				currentSquare = files[col-1] + ranks[start_rank-1]
				
				# Single move forward
				if start_rank < 8:
					newSquare = files[col-1] + ranks[start_rank]
					moves.append(currentSquare + newSquare)
				
				# Double move from rank 2
				if start_rank == 2:
					newSquare = files[col-1] + ranks[start_rank+1]
					moves.append(currentSquare + newSquare)
				
				# Captures (diagonal)
				for offset in [-1, 1]:
					if 1 <= col + offset <= 8:
						if start_rank < 8:
							newSquare = files[col+offset-1] + ranks[start_rank]
							moves.append(currentSquare + newSquare)
		
		# Black pawns (move down, ranks 2-7)
		for col in range(1, 9):
			for start_rank in range(2, 8):
				currentSquare = files[col-1] + ranks[start_rank-1]
				
				# Single move forward (down for black)
				if start_rank > 1:
					newSquare = files[col-1] + ranks[start_rank-2]
					moves.append(currentSquare + newSquare)
				
				# Double move from rank 7
				if start_rank == 7:
					newSquare = files[col-1] + ranks[start_rank-3]
					moves.append(currentSquare + newSquare)
				
				# Captures (diagonal)
				for offset in [-1, 1]:
					if 1 <= col + offset <= 8:
						if start_rank > 1:
							newSquare = files[col+offset-1] + ranks[start_rank-2]
							moves.append(currentSquare + newSquare)
		
		return moves
	
	def generateRookMoves(self):
		"""Generate all rook moves (horizontal and vertical)."""
		moves = []
		files = ['a','b','c','d','e','f','g','h']
		ranks = [str(x) for x in range(1,9)]
		
		rows = list(range(1,9))
		cols = list(range(1,9))
		
		for col in cols:
			for row in rows:
				currentSquare = files[col-1] + ranks[row-1]
				
				# Horizontal moves (same row)
				for newCol in cols:
					if newCol != col:
						newSquare = files[newCol-1] + ranks[row-1]
						moves.append(currentSquare + newSquare)
				
				# Vertical moves (same col)
				for newRow in rows:
					if newRow != row:
						newSquare = files[col-1] + ranks[newRow-1]
						moves.append(currentSquare + newSquare)
		
		return moves
	
	def generateBishopMoves(self):
		"""Generate all bishop moves (diagonal)."""
		moves = []
		files = ['a','b','c','d','e','f','g','h']
		ranks = [str(x) for x in range(1,9)]
		
		rows = list(range(1,9))
		cols = list(range(1,9))
		
		for col in cols:
			for row in rows:
				currentSquare = files[col-1] + ranks[row-1]
				
				# All 4 diagonal directions
				for direction in ['ne','se','sw','nw']:
					for numSquares in range(1,8):
						if direction == 'ne':
							newRow = row + numSquares
							newCol = col + numSquares
						elif direction == 'se':
							newRow = row - numSquares
							newCol = col + numSquares
						elif direction == 'sw':
							newRow = row - numSquares
							newCol = col - numSquares
						elif direction == 'nw':
							newRow = row + numSquares
							newCol = col - numSquares
						
						if self.isSquareValid(newCol, newRow):
							newSquare = files[newCol-1] + ranks[newRow-1]
							moves.append(currentSquare + newSquare)
		
		return moves
	
	def generateKingMoves(self):
		"""Generate all king moves (one square in any direction)."""
		moves = []
		files = ['a','b','c','d','e','f','g','h']
		ranks = [str(x) for x in range(1,9)]
		
		rows = list(range(1,9))
		cols = list(range(1,9))
		
		for col in cols:
			for row in rows:
				currentSquare = files[col-1] + ranks[row-1]
				
				# All 8 directions (one square)
				for dcol in [-1, 0, 1]:
					for drow in [-1, 0, 1]:
						if dcol == 0 and drow == 0:
							continue
						newCol = col + dcol
						newRow = row + drow
						if self.isSquareValid(newCol, newRow):
							newSquare = files[newCol-1] + ranks[newRow-1]
							moves.append(currentSquare + newSquare)
		
		return moves
	
	def generateCastlingMoves(self):
		"""Generate castling moves."""
		moves = []
		# White castling
		moves.append('e1g1')  # Kingside
		moves.append('e1c1')  # Queenside
		# Black castling
		moves.append('e8g8')  # Kingside
		moves.append('e8c8')  # Queenside
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
	