import chess
import numpy as np

class Game(object):

    def __init__(self, history=None, color=True):

        self.history = history or []
        self.img_stack_size = 4
        self.child_visits = []
        self.num_actions = 4672
        self.board = chess.Board()
        self.color = color

    def terminal(self):
        # Game specific termination rules.
        return self.board.is_game_over()
        

    def terminal_value(self, to_play):
        result = self.board.result()
        
        if result == '1-0' and self.color:
            return 1 
        elif result == '0-1' and not self.color:
            return 1 
        elif result == '1/2-1/2':
            return 0
        else:
            return -1 

    def legal_actions(self):
        legal_action_generator = self.board.generate_legal_moves()
        actions = [action.uci() for action in legal_action_generator]
        return actions

    def clone(self):
        return Game(list(self.history))

    def apply(self, action):
        self.board.push(chess.Move.from_uci(action))
        self.history.append(action)

    def store_search_statistics(self, root):
        sum_visits = sum(child.visit_count for child in root.children.itervalues())
        self.child_visits.append([
            root.children[a].visit_count / sum_visits if a in root.children else 0
            for a in range(self.num_actions)
        ])

    def make_image(self, state_index: int):
        # Game specific feature planes.
        # Image is 8x8x(MT+L) (8x8x18)

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
        image[:,:,idx+4] = np.ones((8,8))*self.color

        return image

    def make_target(self, state_index: int):
        return (self.terminal_value(state_index % 2),
                self.child_visits[state_index])

    def to_play(self):
        if self.board.turn == self.color:
            return True
        else:
            return False
            
