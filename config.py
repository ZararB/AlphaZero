class Config(object):

  def __init__(self):

    ### Self-Play
    self.num_actors = 1
    self.num_games_per_epoch = 5

    self.num_sampling_moves = 30
    self.max_moves = 512  # for chess and shogi, 722 for Go.
    self.num_simulations = 800

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
    self.batch_size = 4096

    self.weight_decay = 1e-4
    self.momentum = 0.9
    # Schedule for chess and shogi, Go starts at 2e-2 immediately.
    self.learning_rate_schedule = {
        0: 2e-1,
        100e3: 2e-2,
        300e3: 2e-3,
        500e3: 2e-4
    }

    self.moveDict = self.generateMoveDictionary()


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
                moveDict[uciMove] = moveKey
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
                moveDict[uciMove] = moveKey
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
                moveDict[uciMove] = moveKey
                moveKey += 1 

              
      return moveDict 
      