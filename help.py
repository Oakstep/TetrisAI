from tetris import LocalTetris
from ai2 import TetrisAgent
import random
import numpy as np
import random
import matplotlib.pyplot as plt
import time

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
        self.draw_grid(self.tetris_env.grid, self.axs[0], title="Tetris Grid (Game State)")
        self.draw_piece(self.tetris_env.next_piece, self.axs[2], title="Next Piece")
        self.draw_piece(self.tetris_env.current_piece, self.axs[1], title="Current Piece")
        
        plt.pause(0.01)  # Animation delay


class RandomPlayer:
    def __init__(self, tetris_env: LocalTetris, visualizer: TetrisVisualizer):
        self.tetris_env = tetris_env
        self.visualizer = visualizer

    def play(self):
        while not self.tetris_env.game_over:
            legal_actions = self.tetris_env.get_next_states().keys()
            print(self.tetris_env.current_piece.name)
            print(list(legal_actions))
            action = random.choice(list(legal_actions))
            state, reward, game_over = self.tetris_env.step(action[0], action[1])
            print(f"Action: {action}, Reward: {reward}")
            self.visualizer.visualize()
            time.sleep(10)
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

def test_agent(agent, simulator:LocalTetris, visualizer:TetrisVisualizer, episodes=1):
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
def test_agent_2():
    env = LocalTetris()
    env.reset()
    agent = TetrisAgent(env.get_state_size(), modelFile='tetris_model.keras', epsilon=0)
    visualizer = TetrisVisualizer(env)
    done = False

    while not done:
        visualizer.visualize()
        print(env.current_piece.name)
        next_states = {tuple(v): k for k, v in env.get_next_states().items()}
        best_state = agent.best_state(next_states.keys())
        best_action = next_states[best_state]
        print(best_action)
        _, reward, done = env.step(best_action)
        visualizer.visualize()
        print(reward)
        time.sleep(0.01)
#local_tetris = LocalTetris()
#agent = AIPart(load=True)
#visualizer = TetrisVisualizer(local_tetris)

#run_random()
#test_agent(agent, local_tetris, visualizer, episodes=3) 
#test_agent_2()