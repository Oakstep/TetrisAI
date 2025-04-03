from tetris import LocalTetris
import random
import numpy as np
import random
import matplotlib.pyplot as plt
import time

class TetrisVisualizer:
    def __init__(self, tetris_env):
        self.tetris_env = tetris_env
        self.fig, self.axs = plt.subplots(1, 3, figsize=(12, 6))  # Initialize the figure once
        plt.subplots_adjust(wspace=0.5)  # Space out subplots

    def draw_grid(self, grid, ax, title="Tetris Grid"):
        ax.clear()  # Clear the axes to prevent overlapping visuals
        ax.imshow(grid, cmap="Greys", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)

    def draw_piece(self, piece, ax, title):
        ax.clear()  # Clear the axes for fresh visuals
        if piece is None:  # Handle empty hold or queue
            grid = np.zeros((4, 4))  # Show an empty grid
        else:
            rotation = piece.spawn_rotation  # Use the spawn rotation
            grid = np.zeros((4, 4), dtype=int)  # Center the piece
            piece_height, piece_width = rotation.shape
            grid[:piece_height, :piece_width] = rotation
        ax.imshow(grid, cmap="Greys", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)

    def visualize(self):
        # Update each subplot
        self.draw_grid(self.tetris_env.grid, self.axs[1], title="Tetris Grid (Game State)")
        self.draw_piece(self.tetris_env.held_piece, self.axs[0], title="Held Piece")
        self.draw_piece(self.tetris_env.queue[0], self.axs[2], title="Next Piece in Queue")
        
        # Refresh the figure
        plt.pause(0.5)  # Animation delay


class RandomPlayer:
    def __init__(self, tetris_env: LocalTetris, visualizer: TetrisVisualizer):
        self.tetris_env = tetris_env

    def play(self):
        while not self.tetris_env.game_over:
            legal_actions = self.tetris_env.get_legal_actions()
            action = random.choice(legal_actions)  # Pick a random action
            state, reward, game_over = self.tetris_env.step(action)
            print(f"Action: {action}, Reward: {reward}")
            visualizer.visualize()
            time.sleep(0.5)  # Slow down for visualization
            if game_over:
                print("Game Over!")
                print(self.tetris_env.grid)
                break

local_tetris = LocalTetris()
local_tetris.reset()

visualizer = TetrisVisualizer(local_tetris)
visualizer.visualize()
player = RandomPlayer(local_tetris, visualizer)

# Play and visualize
player.play()