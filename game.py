import chess
import numpy as np
from config import Config 


class Game(object):

    def __init__(self, history=None, historyUCI=None, color=True):
        self.config  = Config()
        self.history = history or []
        self.historyUCI = historyUCI or []
        self.img_stack_size = 4
        self.child_visits = [] #TODO Figure out exactly what this is 
        self.num_actions = self.config.num_actions
        self.board = chess.Board()
        self.color = color

    def terminal(self):
        return self.board.is_game_over()
        

    def terminal_value(self, state_index=None):
        
        result = self.board.result()

        state_index = state_index or len(history)
        
        if result == '1-0' and self.color and state_index % 2 == 0:
            return 1 
        elif result == '0-1' and not self.color and not state_index % 2 == 0:
            return 1 
        elif result == '1/2-1/2':
            return 0
        else:
            return -1 

    def make_target(self, state_index: int):
        #TODO Fix bug, terminal value depends on whether it's agent's turn or not 

        return (np.array(self.termial_value(state_index)),
                np.array(self.child_visits[state_index]).reshape(1, self.num_actions))

    def legal_actions(self):
        legal_action_generator = self.board.generate_legal_moves()
        moveIndices = [self.config.moveList.index(action.uci()) for action in legal_action_generator]
        return moveIndices

    def clone(self):
        cloned_game = Game(list(self.history), list(self.historyUCI))
        cloned_game.board = self.board.copy()
        return cloned_game

    def apply(self, moveIdx):
        move = self.config.moveList[moveIdx]
        self.board.push(chess.Move.from_uci(move))
        self.history.append(moveIdx)
        self.historyUCI.append(move)

    def store_search_statistics(self, root):
        sum_visits = sum(child.visit_count for _, child in root.children.items())
        self.child_visits.append([
            root.children[a].visit_count / sum_visits if a in root.children else 0
            for a in range(self.num_actions)
        ])

    def make_image(self, state_index=None):
        # Game specific feature planes.
        # Image is 8x8x(MT+L) (8x8x18)
        
        to_play = self.to_play(state_index)
        pieces = chess.PIECE_TYPES 
        colors = chess.COLORS
        addInfo = 6 # Castling rights + opp Castling rights + repitition + color 
        num_planes = 12 + addInfo
        image = np.zeros((8,8, num_planes))

        for color in colors:
            for piece in pieces:
                idx = piece - 1 if color else 7 + piece
                image[:,:,idx] = np.array(self.board.pieces(piece, color).tolist()).reshape(8,8)*1

            image[:,:,idx+1] = np.ones((8,8))*self.board.has_queenside_castling_rights(color)
            image[:,:,idx+2] = np.ones((8,8))*self.board.has_kingside_castling_rights(color)

        num_repetitions = [x if self.board.is_repetition(x) == True else 0 for x in range(4)]
        image[:,:, idx+3] = np.ones((8,8))*np.argmax(np.array(num_repetitions))
        image[:,:,idx+4] = np.ones((8,8))*to_play

        return image

    def to_play(self, state_index=None):

        state_index = state_index or len(self.history)  

        if self.color:
            if state_index % 2 == 0:
                return True
            else:
                return False

        else:
            if state_index % 2 == 0:
                return False
            else:
                return True 
        
