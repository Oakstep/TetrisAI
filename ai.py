import cv2
import numpy as np
import win32gui
import win32api
import win32con
import mss
import mss.tools
import time
import tensorflow as tf
from keras._tf_keras.keras import Sequential, Input, Model
from keras._tf_keras.keras.layers import Dense, Flatten, Concatenate



def show(img):
    cv2.imshow("h", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
        return img

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
        return img
    def processSS(self, img):
        if self.x is None:
            self.locate_board(img)
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), (0, 255, 0), 2)
        img[self.y:self.y+self.h, self.x:self.x+self.w] = self.board(img[self.y:self.y+self.h, self.x:self.x+self.w])
        img[self.y:self.y+self.h, self.x+self.w:self.x+self.w+int(0.55*self.w)] = self.queue(img[self.y:self.y+self.h, self.x+self.w:self.x+self.w+int(0.55*self.w)])
        #cv2.imwrite("res_board.png", img)
        return img
    def held(self, img):
        for key in self.pieces:
            pass

class SimActions:
    sct = mss.mss()
    monitor = None
    def __init__(self):
        windows_list = []
        toplist = []
        def enum_win(hwnd, result):
            win_text = win32gui.GetWindowText(hwnd)
            windows_list.append((hwnd, win_text))
        win32gui.EnumWindows(enum_win, toplist)
        self.game_hwnd = 0
        for (hwnd, win_text) in windows_list:
            if "Jstris" in win_text:
                self.game_hwnd = hwnd
                window_rect = win32gui.GetWindowRect(hwnd)
                l, t, r, b = window_rect
                width = r - l
                height = b - t
                self.monitor = {"top": t, "left": l, "width": width, "height": height}
    def clockwise(self):
        win32api.SendMessage(self.game_hwnd, win32con.WM_KEYDOWN, win32con.VK_UP)
        win32api.SendMessage(self.game_hwnd, win32con.WM_KEYUP, win32con.VK_UP)
    def c_clockwise(self):
        win32api.SendMessage(self.game_hwnd, win32con.WM_KEYDOWN, ord('z'), 0)
        win32api.SendMessage(self.game_hwnd, win32con.WM_CHAR, ord('z'), 0)
        win32api.SendMessage(self.game_hwnd, win32con.WM_KEYUP,  ord('z'), 0)
    def flip(self):
        win32api.SendMessage(self.game_hwnd, win32con.WM_KEYDOWN, ord('a'), 0)
        win32api.SendMessage(self.game_hwnd, win32con.WM_CHAR, ord('a'), 0)
        win32api.SendMessage(self.game_hwnd, win32con.WM_KEYUP,  ord('a'), 0)
    def down(self, t):
        win32api.SendMessage(self.game_hwnd, win32con.WM_KEYDOWN, win32con.VK_DOWN)
        time.sleep(t)
        win32api.SendMessage(self.game_hwnd, win32con.WM_KEYUP, win32con.VK_DOWN)
    def right(self):
        win32api.SendMessage(self.game_hwnd, win32con.WM_KEYDOWN, win32con.VK_RIGHT)
        win32api.SendMessage(self.game_hwnd, win32con.WM_KEYUP, win32con.VK_RIGHT)
    def left(self):
        win32api.SendMessage(self.game_hwnd, win32con.WM_KEYDOWN, win32con.VK_LEFT)
        win32api.SendMessage(self.game_hwnd, win32con.WM_KEYUP, win32con.VK_LEFT)
    def hard_drop(self):
        win32api.SendMessage(self.game_hwnd, win32con.WM_KEYDOWN, win32con.VK_SPACE)
        win32api.SendMessage(self.game_hwnd, win32con.WM_KEYUP, win32con.VK_SPACE)
    def hold(self):
        win32api.SendMessage(self.game_hwnd, win32con.WM_KEYDOWN, ord('c'), 0)
        win32api.SendMessage(self.game_hwnd, win32con.WM_CHAR, ord('c'), 0)
        win32api.SendMessage(self.game_hwnd, win32con.WM_KEYUP,  ord('c'), 0)
    def stream(self, tetris: Tetris):
        while True:
            sct_img = self.sct.grab(self.monitor)
            img = np.array(sct_img)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            fresh_frame = img.copy()
            res = tetris.processSS(fresh_frame)
            cv2.imshow('screen', res)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                cv2.destroyAllWindows()
                break
            time.sleep(0.01)

class AIPart:
    def __init__(self, simActions: SimActions):
        self.model = self.create_model()  # Build the neural network model
        self.memory = []  # Replay memory for storing experiences
        self.gamma = 0.95  # Discount factor for future rewards
        self.epsilon = 1.0  # Exploration-exploitation balance
        self.epsilon_min = 0.01  # Minimum exploration probability
        self.epsilon_decay = 0.995  # Decay rate for epsilon
        self.simActions = simActions

    def create_model(self):
        grid_input = Input(shape=(20, 10), name="grid_input")
        flat_grid = Flatten()(grid_input)  # Flatten the board into a 1D array

        current_piece_input = Input(shape=(7,), name="current_piece_input")  # Assume 7 possible pieces

        held_piece_input = Input(shape=(7,), name="held_piece_input")  # Assume 7 possible pieces

        next_pieces_input = Input(shape=(5, 7), name="next_pieces_input")  # 5 pieces, 7 types each
        flat_next_pieces = Flatten()(next_pieces_input)  # Flatten the 5x7 array
        combined = Concatenate()([flat_grid, current_piece_input, held_piece_input, flat_next_pieces])

        dense1 = Dense(128, activation="relu")(combined)
        dense2 = Dense(128, activation="relu")(dense1)

        output = Dense(8, activation="linear", name="q_values")(dense2)

        model = Model(
            inputs=[grid_input, current_piece_input, held_piece_input, next_pieces_input],
            outputs=output,
        )
        model.compile(optimizer="adam", loss="mse")
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def act(self, grid, current_piece, held_piece, next_pieces):
        inputs = {
            "grid_input": grid.reshape(1, 20, 10),
            "current_piece_input": current_piece.reshape(1, 7),
            "held_piece_input": held_piece.reshape(1, 7),
            "next_pieces_input": next_pieces.reshape(1, 5, 7),
        }
        if np.random.rand() <= self.epsilon:
            return np.random.choice(8)  # exploration
        q_values = self.model.predict(inputs)
        return np.argmax(q_values[0])  # exploitation
    def perform_action(self, action):
        if action == 0:  # Left movement
            self.simActions.left()
        elif action == 1:  # Right movement
            self.simActions.right()
        elif action == 2:  # Soft drop
            self.simActions.down()
        elif action == 3:  # Hard drop
            self.simActions.hard_drop()
        elif action == 4:  # Hold
            self.simActions.hold()
        elif action == 5:  # Clockwise rotation
            self.simActions.clockwise()
        elif action == 6:  # Counterclockwise rotation
            self.simActions.c_clockwise()
        elif action == 7:  # 180-degree rotation
            self.simActions.flip()
        #return self.calculate_reward()
    def load(self, name):
        self.model = tf.keras.models.load_model(name)

    def save(self, name):
        self.model.save(name)

print(tf.version)
temp = SimActions()
ai = AIPart(temp)
temp2 = Tetris()
temp.stream(temp2)
sct_img = temp.sct.grab(temp.monitor)
#temp2.locate_board(np.array(sct_img))
#temp2.board(cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR))
cv2.imshow('screen', temp2.processSS(cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)))
cv2.waitKey(0)
cv2.destroyAllWindows()
