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
        self.board.push()
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
        color = chess.Colors
        addInfo = 5 # Castling rights + opp Castling rights + repitition 
        num_planes = 12 + addInfo
        image = np.zeros((8,8, num_planes))


        for color in chess.Colors:

            if color: 
                for piece in pieces:
                    idx = piece - 1 
                    image[8,8,idx] = np.array(board.pieces(piece, color).tolist()).reshape(8,8)*1

                image[8,8,idx+1] = np.ones((8,8))*has_queenside_castling_rights(color)
                image[8,8,idx+2] = np.ones((8,8))*has_kingside)castling_rights(color)

            else:

                for piece in pieces:
                    idx = 7 + piece  
                    image[8,8,idx] = np.array(board.pieces(piece, color).tolist()).reshape(8,8)*1

                image[8,8,idx+1] = np.ones((8,8))*has_queenside_castling_rights(color)
                image[8,8,idx+2] = np.ones((8,8))*has_kingside)castling_rights(color)

            num_repetitions = [1 if chess.is_repetition(x) == True else 0 for x in range(4)]
            image[8,8, idx+3] = np.ones((8,8))*np.argmax(np.array(num_repetitions))


        # Add castling rights to image 
        # Add repitition count to image
        # Add player color 


        return image

    def make_target(self, state_index: int):
        return (self.terminal_value(state_index % 2),
                self.child_visits[state_index])

    def to_play(self):
        return len(self.history) % 2
