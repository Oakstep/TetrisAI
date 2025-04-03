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
        self.rotations = rotations  # Dictionary of rotations
        self.spawn_rotation = rotations["spawn"]  # Default spawn rotation
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
        self.can_hold = True  # Boolean to track hold eligibility
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
        self.current_piece = self.get_next_piece()  # Assign a `TetrisPiece` object
        self.current_rotation_key = "spawn"  # Reset rotation to spawn
        self.held_piece = None  # No piece held at the start
        self.queue = [self.get_next_piece() for _ in range(5)]
        self.game_over = False
        self.can_hold = True  # Reset hold eligibility
        return self.get_state()  # Return the initial state

    def step(self, action):
        # Apply action
        if action["type"] == "hold":
            if self.can_hold:  # Only allow holding if permitted
                self.can_hold = False  # Disable further holding until piece is placed
                if self.held_piece:  # Swap current and held pieces
                    self.current_piece, self.held_piece = self.held_piece, self.current_piece
                    self.current_rotation_key = "spawn"
                else:  # Place current piece into hold
                    self.held_piece = self.current_piece
                    self.current_piece = self.queue.pop(0)
                    self.current_rotation_key = "spawn"
                    self.queue.append(self.get_next_piece())
                reward = self.clear_lines()
            else:
                return self.get_state(), -1, False  # Penalize illegal hold
        else:
            rotation_key = action["rotation"]
            column = action["column"]
            valid = self.place_piece(rotation_key, column)
            if not valid:
                self.game_over = True
                return self.get_state(), -10, True  # Penalty for invalid placement
            reward = self.clear_lines()
            self.current_piece = self.queue.pop(0)  # Transition to the next piece
            self.current_rotation_key = "spawn"  # Reset rotation
            self.queue.append(self.get_next_piece())  # Replenish queue
            self.can_hold = True  # Allow holding again after placing a piece
        
        self.game_over = self.is_terminal_state()
        return self.get_state(), reward, self.game_over

    def place_piece(self, rotation, column):
        piece_shape = self.current_piece.rotations[rotation]
        piece_height, piece_width = piece_shape.shape
        
        # Ensure the piece doesn't go out of bounds
        if column + piece_width > 10:
            print(f"Invalid column: {column}")
            return False
        
        # Start from the top and drop the piece until it lands
        for row in range(20 - piece_height + 1):
            # Check for collision at the next row
            if self.is_collision(row + 1, column, piece_shape) or row + 1 + piece_height > 20:
                # Place the piece at the current row (the last valid position)
                self.grid[row:row+piece_height, column:column+piece_width] += piece_shape
                return True
        
        # If no valid position is found (unlikely), return False
        return False



    def is_collision(self, row, column, piece_shape):
        piece_height, piece_width = piece_shape.shape
        
        # Ensure we're within bounds
        if row + piece_height > 20 or column + piece_width > 10:
            return True
        
        # Check for overlap with existing blocks
        grid_section = self.grid[row:row+piece_height, column:column+piece_width]
        return np.any(grid_section + piece_shape > 1)



    def clear_lines(self):
        lines_cleared = 0
        for row in range(20):
            if all(self.grid[row]):
                lines_cleared += 1
                self.grid[1:row+1] = self.grid[:row]
                self.grid[0] = np.zeros(10)
        return lines_cleared ** 2  # Quadratic reward for line clears

    def is_terminal_state(self):
        return any(self.grid[0])  # Game over if top row is filled

    def get_state(self):
        print(f"Grid: \n{self.grid}")
        print(f"Current piece: {self.current_piece.name if self.current_piece else None}")
        print(f"Held piece: {self.held_piece.name if self.held_piece else None}")
        print(f"Queue: {[piece.name for piece in self.queue]}")
        print(f"Can hold: {self.can_hold}")
        return {
            "grid": self.grid.copy(),
            "current_piece": self.current_piece,
            "held_piece": self.held_piece,
            "queue": self.queue,
            "can_hold": self.can_hold
        }


    def get_legal_actions(self):
        legal_actions = []
        # Iterate over all possible rotations for the current piece
        for rotation_key in self.current_piece.rotations.keys():
            # Iterate over valid columns for this rotation
            for column in range(10 - self.current_piece.rotations[rotation_key].shape[1] + 1):
                if not self.is_collision(0, column, self.current_piece.rotations[rotation_key]):
                    legal_actions.append({"type": "place", "rotation": rotation_key, "column": column})
        # Add the hold action if holding is allowed
        if self.can_hold:
            legal_actions.append({"type": "hold"})
        return legal_actions

