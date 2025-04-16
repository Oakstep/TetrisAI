from ai4 import TetrisAgent
from tetris import LocalTetris
from help import TetrisVisualizer

env = LocalTetris()
env.reset()
agent = TetrisAgent(env.get_state_size(), modelFile='sample.keras', epsilon=0)
visualizer = TetrisVisualizer(env)
done = False

while not done:
    #print(env.current_piece.name)
    next_states = {tuple(v): k for k, v in env.get_next_states().items()}
    best_state = agent.act(next_states.keys())
    best_action = next_states[best_state]
    print(best_action)
    _, reward, done = env.step(best_action[0], best_action[1])
    visualizer.visualize()
    print(env.grid)
    #time.sleep(0.001)