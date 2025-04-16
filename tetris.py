import random
import copy
import numpy as np

class TetrisPiece:
    def __init__(self, name, rotations):
        self.name = name
        self.rotations = rotations
        self.spawn_rotation = rotations["spawn"]
        self.width = self.spawn_rotation.shape[1]  # Width of the spawn state
        self.height = self.spawn_rotation.shape[0]  # Height of the spawn state
class LocalTetris:
    def __init__(self):
        self.grid = np.zeros((20, 10), dtype=int)
        self.pieces = self.define_tetris_pieces()
        self.bag = self.generate_bag()
        self.current_piece = self.get_next_piece()
        self.current_rotation_key = "spawn"
        self.next_piece = self.get_next_piece()
        #self.held_piece = None
        self.game_over = False
        #self.can_hold = True
        self.score = 0
        self.cleared_lines = 0
    
    def define_tetris_pieces(self):
        pieces = {
            "I": TetrisPiece("I", {
                "spawn": np.array([[1, 1, 1, 1]]),  # Default horizontal
                "clockwise": np.array([[1], [1], [1], [1]]),  # Vertical
                "counterclockwise": np.array([[1], [1], [1], [1]]),  # Same as clockwise for "I"
                "flip": np.array([[1, 1, 1, 1]])  # Same as spawn
            }),
            "J": TetrisPiece("J", {
                "spawn": np.array([[1, 0, 0], [1, 1, 1]]),
                "clockwise": np.array([[1, 1], [1, 0], [1, 0]]),
                "counterclockwise": np.array([[0, 1], [0, 1], [1, 1]]),
                "flip": np.array([[1, 1, 1], [0, 0, 1]])
            }),
            "L": TetrisPiece("L", {
                "spawn": np.array([[0, 0, 1], [1, 1, 1]]),
                "clockwise": np.array([[1, 1], [0, 1], [0, 1]]),
                "counterclockwise": np.array([[1, 0], [1, 0], [1, 1]]),
                "flip": np.array([[1, 1, 1], [1, 0, 0]])
            }),
            "O": TetrisPiece("O", {
                "spawn": np.array([[1, 1], [1, 1]]),  # Static (no rotations)
                "clockwise": np.array([[1, 1], [1, 1]]),
                "counterclockwise": np.array([[1, 1], [1, 1]]),
                "flip": np.array([[1, 1], [1, 1]])
            }),
            "S": TetrisPiece("S", {
                "spawn": np.array([[0, 1, 1], [1, 1, 0]]),
                "clockwise": np.array([[1, 0], [1, 1], [0, 1]]),
                "counterclockwise": np.array([[1, 0], [1, 1], [0, 1]]),
                "flip": np.array([[0, 1, 1], [1, 1, 0]])
            }),
            "Z": TetrisPiece("Z", {
                "spawn": np.array([[1, 1, 0], [0, 1, 1]]),
                "clockwise": np.array([[0, 1], [1, 1], [1, 0]]),
                "counterclockwise": np.array([[0, 1], [1, 1], [1, 0]]),
                "flip": np.array([[1, 1, 0], [0, 1, 1]])
            }),
            "T": TetrisPiece("T", {
                "spawn": np.array([[0, 1, 0], [1, 1, 1]]),
                "clockwise": np.array([[1, 0], [1, 1], [1, 0]]),
                "counterclockwise": np.array([[0, 1], [1, 1], [0, 1]]),
                "flip": np.array([[1, 1, 1], [0, 1, 0]])
            })
        }
        return pieces
    def generate_bag(self):
        pieces = list(self.pieces.keys())  # "I", "J", "L", "O", "S", "Z", "T"
        random.shuffle(pieces)
        return pieces

    def get_next_piece(self):
        if not self.bag:
            self.bag = self.generate_bag()
        piece_name = self.bag.pop()
        return self.pieces[piece_name]

    def reset(self):
        self.grid = np.zeros((20, 10), dtype=int)
        self.bag = self.generate_bag()
        self.current_piece = self.get_next_piece()
        self.current_rotation_key = "spawn" 
        self.next_piece = self.get_next_piece()
        #self.held_piece = None 
        self.game_over = False
        #self.can_hold = True
        self.score = 0
        self.cleared_lines = 0
        return self.get_state()


    def step(self, x, rotation):
        rotation_map = {
            0: "spawn",
            1: "clockwise",
            2: "flip",
            3: "counterclockwise"
        }
        self.place_piece(rotation_map[rotation], x)
        self.game_over = self.is_terminal_state()
        reward = self.reward() 
        reward = reward - 5 if self.game_over else 0
        self.current_piece = self.next_piece
        self.current_rotation_key = "spawn"
        self.next_piece = self.get_next_piece()
        return self.get_state(), reward, self.game_over 

    def place_piece(self, rotation, column):
        piece_shape = self.current_piece.rotations[rotation]
        piece_height, piece_width = piece_shape.shape
        
        if column + piece_width > 10:
            print(f"Invalid column: {column}")
            return False
        for row in range(20 - piece_height + 1):
            if self.is_collision(row + 1, column, piece_shape) or row + 1 + piece_height > 20:
                self.grid[row:row+piece_height, column:column+piece_width] += piece_shape
                return True
        return False

    def is_collision(self, row, column, piece_shape):
        piece_height, piece_width = piece_shape.shape
        
        if row + piece_height > 20 or column + piece_width > 10:
            return True
        grid_section = self.grid[row:row+piece_height, column:column+piece_width]
        return np.any(grid_section + piece_shape > 1)

    def is_terminal_state(self):
        return any(self.grid[0])
    
    def get_state(self):
        current_piece = self.one_hot_encode(self.current_piece)
        next_piece = self.one_hot_encode(self.next_piece)
        #held_piece = self.one_hot_encode(self.held_piece)
        #can_hold = 1 if self.can_hold else 0
        holes = self.calculate_holes()
        bumpiness = self.get_bumpiness()
        height = self.get_total_height()
        return [self.cleared_lines, holes, bumpiness, height]

    def get_legal_actions(self):
        rotation_map = {
            "spawn": 0,
            "clockwise": 1,
            "flip": 2,
            "counterclockwise": 3
        }
        if self.game_over:
            return []
        legal_actions = []
        for rotation_key in self.current_piece.rotations.keys():
            piece_width = self.current_piece.rotations[rotation_key].shape[1]
            for column in range(10):
                if column+piece_width > 10:
                    continue
                elif not self.is_collision(0, column, self.current_piece.rotations[rotation_key]):
                    legal_actions.append((column, rotation_map[rotation_key]))
        return legal_actions

    def get_next_states(self):
        next_states = {}
        legal_actions = self.get_legal_actions()
        backup = self.snapshot_state()
        for action in legal_actions:
            _, _, _ = self.step(action[0], action[1])
            next_states[action] = self.get_state() 
            self.load_state(backup)
        return next_states
    

    def snapshot_state(self):
        return copy.deepcopy({
            "grid": self.grid,
            "current_piece": self.current_piece,
            "current_rotation_key": self.current_rotation_key,
            "next_piece": self.next_piece,
            "game_over": self.game_over,
            "score": self.score,
            "bag": self.bag,
            "lines_cleared": self.cleared_lines
        })
    def load_state(self, state):
        self.grid = copy.deepcopy(state["grid"])
        self.current_piece = copy.deepcopy(state["current_piece"])
        self.current_rotation_key = copy.deepcopy(state["current_rotation_key"])
        self.next_piece = copy.deepcopy(state["next_piece"])
        self.game_over = copy.deepcopy(state["game_over"])
        self.score = copy.deepcopy(state["score"])
        self.bag = copy.deepcopy(state["bag"])
        self.cleared_lines = copy.deepcopy(state["lines_cleared"])

    def one_hot_encode(self, piece: TetrisPiece, all_pieces=["I", "J", "L", "O", "S", "T", "Z"]):
        if piece is not None:
            return all_pieces.index(piece.name)
        return -1  # or len(all_pieces) if you want to reserve an index for "None"

    def calculate_holes(self):
        holes = 0
        for col in range(10):  # Iterate through each column
            column = self.grid[:, col]  # Extract the column
            block_found = False  # Flag to check if we've found a block
            for cell in column:
                if cell > 0:
                    block_found = True  # Start counting holes after the first block
                elif block_found and cell == 0:
                    holes += 1  # Count empty cells below a block
        return holes
    
    def get_state_size(self):
        return 4
    
    def get_game_score(self):
        return self.score
    
    def reward(self):
        lines_cleared = 0
        reward = 0
        for row in range(20):
            if all(self.grid[row]):
                lines_cleared += 1
                self.grid[1:row+1] = self.grid[:row]
                self.grid[0] = np.zeros(10)
        if lines_cleared == 1:
            reward += 100
        elif lines_cleared == 2:
            reward += 200
        elif lines_cleared == 3:
            reward += 400
        elif lines_cleared == 4:
            reward += 1200  # Tetris
        self.score += reward
        self.cleared_lines = lines_cleared
        return reward
    def get_column_heights(self):
        heights = []
        for col in range(self.grid.shape[1]):
            column = self.grid[:, col]
            filled_indices = np.where(column > 0)[0]
            if len(filled_indices) == 0:
                heights.append(0)
            else:
                # Height = total rows - first filled row index
                height = self.grid.shape[0] - filled_indices[0]
                heights.append(height)
        return heights
    def get_bumpiness(self):
        heights = self.get_column_heights()
        return sum(abs(heights[i] - heights[i + 1]) for i in range(len(heights) - 1))
    def get_total_height(self):
        return sum(self.get_column_heights())

