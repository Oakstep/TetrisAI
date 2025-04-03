import numpy as np
import time
import cv2
from tetris import Tetris
from controller import SimActions
import mss
import mss.tools
from ai import AIPart


class Main: 
    sct = None
    monitor = None
    def __init__(self, controller: SimActions, agent: AIPart, tetris: Tetris):
        self.controller = controller
        self.monitor = controller.monitor
        self.sct = controller.sct
        self.tetris = tetris
        self.agent = agent
    
    def one_hot(self, piece):
        """
        Convert a Tetris piece letter (e.g., 'l', 'i', 'j', 'o', 's', 'z', 't')
        into a one-hot vector of length 7.
        """
        mapping = {'l': 0, 'i': 1, 'j': 2, 'o': 3, 's': 4, 'z': 5, 't': 6}
        onehot = np.zeros((7,), dtype=np.float32)
        if piece and piece.lower() in mapping:
            onehot[mapping[piece.lower()]] = 1.0
        return onehot

    def one_hot_queue(self, queue, expected_length=5):
        """
        Convert a list of piece letters into a numpy array of one-hot vectors.
        Pads with zeros if the queue is shorter than expected_length.
        """
        onehot_list = [self.one_hot(p) for p in queue] if queue else []
        # Pad with zeros if necessary.
        while len(onehot_list) < expected_length:
            onehot_list.append(np.zeros((7,), dtype=np.float32))
        return np.array(onehot_list, dtype=np.float32)
    
    def stream(self, max_steps=500, batch_size=32):
        step = 0
        while True:
            sct_img = self.sct.grab(self.monitor)
            img = np.array(sct_img)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            fresh_frame = img.copy()
            res, board_array, queue, held, current = self.tetris.processSS(fresh_frame)
            sim_state = {
                "grid": board_array,  # shape (20,10)
                "current_piece": self.one_hot(current) if current else np.zeros((7,), dtype=np.float32),
                "held_piece": self.one_hot(held) if held else np.zeros((7,), dtype=np.float32),
                "next_pieces": [self.one_hot(p) for p in queue] if queue else []  # variable-length list
            }
            
            replay_state = {
                "grid_input": board_array.reshape(1, 20, 10),
                "current_piece_input": self.one_hot(current).reshape(1, 7) if current else np.zeros((1,7), dtype=np.float32),
                "held_piece_input": self.one_hot(held).reshape(1, 7) if held else np.zeros((1,7), dtype=np.float32),
                "next_pieces_input": self.one_hot_queue(queue, expected_length=5).reshape(1, 5, 7)
            }
            action = self.agent.pick_action(sim_state)
            self.controller.perform_action(action)
            cv2.imshow('screen', res)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                cv2.destroyAllWindows()
                break
            step += 1
            time.sleep(0.01)

actions = SimActions()
main = Main(actions)
#agent = AIPart(actions)
env = Tetris()
main.stream(env)
cv2.waitKey(0)
cv2.destroyAllWindows()
#train_ai(env, agent, episodes=1000, batch_size=32)