from ast import List
from math import inf
from threading import local
import numpy as np
import cv2
import random

class Tetris:
    width = 10
    height = 20
    board_array = np.zeros((20, 10))
    mask = None
    grey = [57,57,57]
    x=y=w=h = None
    pieces = {"i": [15,155,215], #light blue
               "o": [227, 159, 2], #yellow
               "t": [175,41,138], #purple
               "j": [227,91,2], #orange
               "l": [33,65,198], #blue
               "s": [89,177,1], #green
               "z": [215,15,55]} #red
    def __init__(self):
        for key in self.pieces:
            bgr_color = self.pieces[key]
            bgr_color = np.flip(np.array(bgr_color, dtype='uint8'))
            self.pieces[key] = bgr_color
    def queue(self, img):
        arr = []
        for key in self.pieces:
            bgr_color = self.pieces[key]
            mask = cv2.inRange(img, bgr_color, bgr_color)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
            for cnt in contours:
                (x, y, w, h) = cv2.boundingRect(cnt)
                arr.append((key, y))
        arr = sorted(arr, key=lambda x: x[1])
        arr = [item[0] for item in arr]
        return img, arr
    def held(self, img):
        arr = []
        for key in self.pieces:
            bgr_color = self.pieces[key]
            mask = cv2.inRange(img, bgr_color, bgr_color)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
            for cnt in contours:
                (x, y, w, h) = cv2.boundingRect(cnt)
                arr.append((key, y))
        arr = sorted(arr, key=lambda x: x[1])
        arr = [item[0] for item in arr]
        return img, arr[0] if arr else None
    def locate_board(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        max_rectangle = None

        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) == 4:
                area = cv2.contourArea(approx)
                if area > max_area:
                    max_area = area
                    max_rectangle = approx
        if max_rectangle is not None:
            x, y, w, h = cv2.boundingRect(max_rectangle)
            self.x, self.y, self.w, self.h = x,y,w,h
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #cv2.imwrite("res_board.png", img)
        return img
    def board(self, img):
        flag = True
        #if self.mask is None:
        for key in self.pieces:
            bgr_color = self.pieces[key]
            mask = cv2.inRange(img, bgr_color, bgr_color)
            if flag is True:
                self.mask = mask
                flag = False
            else:
                self.mask = cv2.bitwise_or(self.mask, mask)
        result = cv2.bitwise_and(img, img, mask=self.mask)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        arr = []
        for cnt in contours:
                (x, y, w, h) = cv2.boundingRect(cnt)
                arr.append((cnt, y))
        arr = sorted(arr, key=lambda x: x[1])
        arr = [item[0] for item in arr]
        cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
        height, width, channels = img.shape
        cell_width = width/self.width
        cell_height = height/self.height
        for row in range(self.height):
            for col in range(self.width):
                x1 = int(col * cell_width)
                y1 = int(row * cell_height)
                x2 = int((col + 1) * cell_width)
                y2 = int((row + 1) * cell_height)
                cell = result[y1:y2, x1:x2]
                if np.sum(cell > 0) > (cell_width * cell_height * 0.5):
                    self.board_array[row, col] = 1
        return img, self.board_array
    def current(self, img): 
        arr = []
        for key in self.pieces:
            bgr_color = self.pieces[key]
            mask = cv2.inRange(img, bgr_color, bgr_color)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
            for cnt in contours:
                (x, y, w, h) = cv2.boundingRect(cnt)
                arr.append((key, y))
        arr = sorted(arr, key=lambda x: x[1])
        arr = [item[0] for item in arr]
        return img, arr[0] if arr else None
    def processSS(self, img):
        if self.x is None:
            self.locate_board(img)
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), (0, 255, 0), 2)
        img[self.y:self.y+self.h, self.x:self.x+self.w], board_array = self.board(img[self.y:self.y+self.h, self.x:self.x+self.w])
        img[self.y:self.y+self.h, self.x+self.w:self.x+self.w+int(0.55*self.w)], queue = self.queue(img[self.y:self.y+self.h, self.x+self.w:self.x+self.w+int(0.55*self.w)])
        img[self.y:self.y+self.h, self.x-self.w:self.x], held = self.held(img[self.y:self.y+self.h, self.x-self.w:self.x])
        _, current = self.current(img[self.y:self.y+self.h, self.x:self.x+self.w])
        #cv2.imwrite("res_board.png", img)
        return img, board_array, queue, held, current
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
        self.held_piece = None
        self.queue = [self.get_next_piece() for _ in range(5)]
        self.game_over = False
        self.can_hold = True
        self.back_to_back = None
        self.score = 0
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
        self.held_piece = None 
        self.queue = [self.get_next_piece() for _ in range(5)]
        self.game_over = False
        self.can_hold = True
        self.back_to_back = None
        self.score = 0
        return self.get_state()

    def step(self, action):
        # Apply action
        if action["type"] == "hold":
            if self.can_hold:
                self.can_hold = False
                if self.held_piece:
                    self.current_piece, self.held_piece = self.held_piece, self.current_piece
                    self.current_rotation_key = "spawn"
                else:
                    self.held_piece = self.current_piece
                    self.current_piece = self.queue.pop(0)
                    self.current_rotation_key = "spawn"
                    self.queue.append(self.get_next_piece())
                reward = self.reward()
            else:
                return self.get_state(), -1, False
        else:
            rotation_key = action["rotation"]
            column = action["column"]
            valid = self.place_piece(rotation_key, column)
            if not valid:
                self.game_over = True
                return self.get_state(), -10, True
            reward = self.reward()
            self.current_piece = self.queue.pop(0) 
            self.current_rotation_key = "spawn"
            self.queue.append(self.get_next_piece())
            self.can_hold = True
        
        self.game_over = self.is_terminal_state()
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

    def reward(self):
        if(self.is_terminal_state()):
            return -999999
        lines_cleared = 0
        for row in range(20):
            if all(self.grid[row]):
                lines_cleared += 1
                self.grid[1:row+1] = self.grid[:row]
                self.grid[0] = np.zeros(10)
        garbage_rewards = {0: 0, 1: 0, 2: 1, 3: 2, 4: 4}  # Points based on lines cleared
        reward = garbage_rewards[lines_cleared]
        if lines_cleared > 0 and lines_cleared == self.back_to_back:
            reward += 1
        elif lines_cleared > 0:
            self.back_to_back = lines_cleared
        self.score += reward
        reward = (reward*2)**2
        holes = self.calculate_holes()
        if holes == 0:
            reward+=5
        else:
            reward -= holes**2
        penalty_for_height = 0
        for row in range(0, 8):
            penalty_for_height += (sum(self.grid[row]) * (20-row))  
        if penalty_for_height == 0:
            reward+=5
        else:
            reward -= penalty_for_height
        return reward
    
    
    def is_terminal_state(self):
        return any(self.grid[0])
    
    def get_state(self):
        return {
            "grid": np.array(self.grid.copy(), dtype=np.float32).reshape(1, 20, 10),
            "current_piece": np.array(self.one_hot_encode(self.current_piece), dtype=np.float32).reshape(1, 7),
            "held_piece": np.array(self.one_hot_encode(self.held_piece), dtype=np.float32).reshape(1, 7),
            "queue": np.array(
                [self.one_hot_encode(piece) for piece in self.queue], dtype=np.float32
            ).reshape(1, 5, 7),
            "can_hold": np.array([[1 if self.can_hold else 0]], dtype=np.float32),
        }

    def get_legal_actions(self):
        if self.game_over:
            return []
        legal_actions = []
        action_index = 0
        for rotation_key in self.current_piece.rotations.keys():
            piece_width = self.current_piece.rotations[rotation_key].shape[1]
            for column in range(10):
                if column+piece_width > 10:
                    action_index += 1
                    continue
                elif not self.is_collision(0, column, self.current_piece.rotations[rotation_key]):
                    legal_actions.append({
                        "type": "place", 
                        "rotation": rotation_key, 
                        "column": column,
                        "index": action_index
                    })
                    action_index += 1
        if self.can_hold:
            legal_actions.append({"type": "hold", "index": action_index})
            action_index += 1
        return legal_actions

    def get_next_states(self):
        next_states = {}
        if(self.is_terminal_state()):
            return next_states
        legal_actions = self.get_legal_actions()
        grid_backup = self.grid.copy()
        current_piece_backup = self.current_piece
        current_rotation_key_backup = self.current_rotation_key
        held_piece_backup = self.held_piece
        queue_backup = self.queue.copy()
        can_hold_backup = self.can_hold
        game_over_backup = self.game_over
        score_backup = self.score
        for action in legal_actions:
            next_state, reward, _ = self.step(action)
            action_key = tuple(action.items())
            next_states[action_key] = (next_state, reward)

            self.grid = grid_backup
            self.current_piece = current_piece_backup
            self.current_rotation_key = current_rotation_key_backup
            self.held_piece = held_piece_backup
            self.queue = queue_backup
            self.can_hold = can_hold_backup
            self.game_over = game_over_backup
            self.score = score_backup
        return next_states

    def one_hot_encode(self, piece: TetrisPiece, all_pieces=["I", "J", "L", "O", "S", "T", "Z"]):
        encoding = np.zeros(len(all_pieces), dtype=int)
        if piece is not None:
            index = all_pieces.index(piece.name)
            encoding[index] = 1
        return encoding

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
def test():
    tetr = LocalTetris();
    print(f"Grid: {tetr.grid}")
    print(f"Current Piece: {tetr.current_piece.name}")
    print(f"Legal Actions: {tetr.get_legal_actions()}")
#test()