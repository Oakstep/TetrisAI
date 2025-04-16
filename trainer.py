from ai4 import TetrisAgent
from tetris import LocalTetris

env = LocalTetris()
max_episode = 2000
max_steps = 25000
agent = TetrisAgent(env.get_state_size())

episodes = []
rewards = []

current_max = 0
for generation in range(10):
    agent.epsilon = 1.0
    for episode in range(max_episode):
        current_state = env.reset()
        done = False
        steps = 0
        total_reward = 0
        print(f"Running episode: {episode} in Generation: {generation} with Epsilon:{agent.epsilon}")

        while not done and steps < max_steps:
            next_states = env.get_next_states()
            if not next_states:
                break
            best_state = agent.act(next_states.values())
            best_action = next(k for k, v in next_states.items() if v == best_state)
            _, reward, done = env.step(best_action[0], best_action[1])
            total_reward += reward
            agent.add_to_memory(current_state, best_state, reward, done)
            current_state = best_state
            steps += 1

        print("Total reward: " + str(total_reward))
        episodes.append(episode)
        rewards.append(total_reward)
        agent.train()
        agent.save_model()
    agent.save_model(f"generation {generation}.keras")