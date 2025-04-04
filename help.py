from tetris import LocalTetris, Tetris
import random
import numpy as np
import random
import matplotlib.pyplot as plt
import time
from ai import AIPart

class TetrisVisualizer:
    def __init__(self, tetris_env):
        self.tetris_env = tetris_env
        self.fig, self.axs = plt.subplots(1, 3, figsize=(12, 6)) 
        plt.subplots_adjust(wspace=0.5)

    def draw_grid(self, grid, ax, title="Tetris Grid"):
        ax.clear()
        ax.imshow(grid, cmap="Greys", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)

    def draw_piece(self, piece, ax, title):
        ax.clear() 
        if piece is None: 
            grid = np.zeros((4, 4)) 
        else:
            rotation = piece.spawn_rotation 
            grid = np.zeros((4, 4), dtype=int) 
            piece_height, piece_width = rotation.shape
            grid[:piece_height, :piece_width] = rotation
        ax.imshow(grid, cmap="Greys", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)

    def visualize(self):
        self.draw_grid(self.tetris_env.grid, self.axs[1], title="Tetris Grid (Game State)")
        self.draw_piece(self.tetris_env.held_piece, self.axs[0], title="Held Piece")
        self.draw_piece(self.tetris_env.queue[0], self.axs[2], title="Next Piece in Queue")
        
        plt.pause(0.5)  # Animation delay


class RandomPlayer:
    def __init__(self, tetris_env: LocalTetris, visualizer: TetrisVisualizer):
        self.tetris_env = tetris_env
        self.visualizer = visualizer

    def play(self):
        while not self.tetris_env.game_over:
            legal_actions = self.tetris_env.get_legal_actions()
            action = random.choice(legal_actions)
            state, reward, game_over = self.tetris_env.step(action)
            print(f"Action: {action}, Reward: {reward}")
            self.visualizer.visualize()
            time.sleep(0.5)
            if game_over:
                print("Game Over!")
                break

def run_random():
    local_tetris = LocalTetris()
    local_tetris.reset()

    visualizer = TetrisVisualizer(local_tetris)
    #visualizer.visualize()
    player = RandomPlayer(local_tetris, visualizer)

    player.play()

def test_agent(agent:AIPart, simulator:LocalTetris, visualizer:TetrisVisualizer, episodes=1):
    for episode in range(episodes):
        state = simulator.reset() 
        done = False

        print(f"Starting Episode {episode + 1}")

        while not done:
            legal_actions = simulator.get_legal_actions()
            if not legal_actions:
                print("No legal actions available. Ending episode.")
                break

            action = agent.act(state, legal_actions)
            print(f"action: {action}")
            if action is None:
                print("No action selected. Ending episode.")
                break

            next_state, reward, done = simulator.step(action)
            print(reward)
            visualizer.visualize()
            state = next_state
            time.sleep(1)
        print(f"Episode {episode + 1} finished.")
local_tetris = LocalTetris()
agent = AIPart(load=True)
visualizer = TetrisVisualizer(local_tetris)

#run_random()
test_agent(agent, local_tetris, visualizer, episodes=3) 