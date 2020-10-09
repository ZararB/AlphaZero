from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Convolution2D, MaxPooling2D, Dropout
import numpy as np

class Network(object):

	def __init__(self, config=None):
		self.model = Sequential([

			])

	def inference(self, image):
		model_output = self.model.predict(image)
		value  = 5 
		policy = np.random.randn(4672)
		return (model_output[0,0], model_output[0,1])  # Value, Policy

	def policy2dictionary(self, policy):
		policyDict = {}


		return policyDict 

	def generateMoveDictionary(self):
		moveDict = {}
		moveKey = 0 

		colLetters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
	
		for col in range(1,9):
			for row in range(1,9):
				startSquare = colLetters[col-1] + str(row)
				# Queen moves
				for direction in ['n','ne', 'e','se', 's', 'sw', 'w', 'nw']:
					for numSquares in range(1,8):
						if direction == 'n':
							endCol = col 
							endRow = row + numSquares

						elif direction == 'ne':
							endCol = col + numSquares 
							endRow = row + numSquares 

						elif direction == 'e':
							endCol = col + numSquares
							endRow = row

						elif direction == 'se':
							endCol = col + numSquares
							endRow = row - numSquares

						elif direction == 's':
							endCol = col 
							endRow = row - numSquares

						elif direction == 'sw':
							endCol = col - numSquares
							endRow = row - numSquares 

						elif direction == 'w':
							endCol = col - numSquares 
							endRow = row 

						elif direction == 'nw':
							endCol = col - numSquares
							endRow = row + numSquares


						if (endCol >= 1 and endCol <=8) and (endRow >= 1 and endRow <=8):

							endSquare = colLetters[endCol-1] + str(endRow)
							uciMove = startSquare + endSquare
							moveDict[moveKey] = uciMove
							moveKey += 1 

				# Knight moves 
				for move in ['ne', 'nw', 'se', 'sw', 'en', 'es', 'wn', 'ws']:

					if move == 'ne':
						endCol = col + 1
						endRow = row + 2

					elif move == 'nw':
						endCol = col - 1 
						endRow = row + 2 

					elif move == 'se':
						endCol = col + 1 
						endRow = row - 2 

					elif move == 'sw':
						endCol = col - 1 
						endRow = row - 2 
						
					elif move == 'en':
						endCol = col + 2 
						endRow = row + 1

					elif move == 'es':
						endCol = col + 2  
						endRow = row - 1 

					elif move == 'wn':
						endCol = col - 2
						endRow = row + 1 

					elif move == 'ws':
						endCol = col - 2 
						endRow = row - 1

					if (endCol >= 1 and endCol <=8) and (endRow >= 1 and endRow <=8):

							endSquare = colLetters[endCol-1] + str(endRow)
							uciMove = startSquare + endSquare
							moveDict[moveKey] = uciMove
							moveKey += 1     

				# Pawn underpromotions 
				for direction in ['n','ne','se', 's', 'sw', 'nw']:
					for promotionPiece in ['k', 'r', 'b']:
						if direction == 'n':
							endCol = col
							endRow = row + 1 

						elif direction == 'ne':
							endCol = col + 1 
							endRow = row + 1 

						elif direction == 'ne':
							endCol = col + 1 
							endRow = row + 1

						elif direction == 's':
							endCol = col  
							endRow = row - 1 

						elif direction == 'se':
							endCol = col + 1 
							endRow = row - 1 

						elif direction == 'sw':
							endCol = col - 1 
							endRow = row - 1 

						if (endCol >= 1 and endCol <=8) and (endRow >= 1 and endRow <=8):

							endSquare = colLetters[endCol-1] + str(endRow) 
							uciMove = startSquare + endSquare + promotionPiece 
							moveDict[moveKey] = uciMove
							moveKey += 1 

						
		return moveDict 
		

					



					 
					







	def get_weights(self):
		# Returns the weights of this network.
		return []




network = Network()
moveDict = network.generateMoveDictionary()
