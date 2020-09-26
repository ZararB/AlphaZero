import chess

class Game(object):

    def __init__(self, history=None):

        self.history = history or []
        self.img_stack_size = 4
        self.child_visits = []
        self.num_actions = 4672
        self.board = chess.Board()

    def terminal(self):
        # Game specific termination rules.
        pass
        

    def terminal_value(self, to_play):
        pass
    
    def legal_actions(self):
        legal_action_generator = self.board.legal_action_generator()
        actions = [action for action in legal_action_generator]

        return actions

    def clone(self):
        return Game(list(self.history))

    def apply(self, action):
        self.history.append(action)

    def store_search_statistics(self, root):
        sum_visits = sum(child.visit_count for child in root.children.itervalues())
        self.child_visits.append([
            root.children[a].visit_count / sum_visits if a in root.children else 0
            for a in range(self.num_actions)
        ])

    def make_image(self, state_index: int):
        # Game specific feature planes.
        # image is 8x8x(MT + L) where M = number of player piece types + number of opponent piece types
        # pieces = ['b', 'k', 'n', 'p', 'q', 'r', 'B', 'K', 'N', 'P', 'Q', 'R']
        pieces = chess.PIECE_TYPES 
        addInfo = 5 # Castling rights + opp Castling rights + repitition 
        num_planes = num_player_pieces + num_opponent_pieces + addInfo
        image = np.zeros((8,8, 12))

        for piece in pieces:
            for color in colors:
                if color:
                    idx = piece-1
                    image[8,8,idx] = np.array(board.pieces(piece, color).tolist()).reshape(8,8)*1


                else:
                    idx = 5 + piece 
                    image[8,8,idx] = np.array(board.pieces(piece, color).tolist()).reshape(8,8)*1


        # Add castling rights to image 
        # Add repitition count to image


        return image

    def make_target(self, state_index: int):
        return (self.terminal_value(state_index % 2),
                self.child_visits[state_index])

    def to_play(self):
        return len(self.history) % 2
