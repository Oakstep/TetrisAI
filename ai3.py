import numpy as np
import tensorflow as tf
from keras._tf_keras import keras
from keras._tf_keras.keras import Sequential, Input, Model
from keras._tf_keras.keras.layers import Dense, Flatten, Concatenate, InputLayer
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.optimizers import Adam
from collections import deque
from tetris import Tetris, LocalTetris
from controller import SimActions
import random
from tqdm import tqdm
class TetrisAgent:
    def __init__(self, state_size, hidden_layers=[128, 64], activations=['relu', 'relu', 'linear'],
                 loss='mse', lr=0.001,
                 memory_size=20000, batch_size=512, replay_start_size=1500, modelFile=None, epsilon=1, episodes=1000):
        self.action_size = 2
        self.state_size = state_size + 2
        self.gamma = 0.95
        self.epsilon = epsilon
        self.epsilon_min = 0.001
        self.epsilon_decay = epsilon/episodes
        self.batch_size = batch_size
        self.replay_start_size = replay_start_size

        self.memory = deque(maxlen=memory_size)        
        if modelFile is not None:
            self.model = load_model(modelFile)
        else:
            self.model = self.create_model(
            input_size=state_size + 2,
            hidden_layers=hidden_layers,
            activations=activations,
            loss=loss,
            learning_rate=lr
        )
    def create_model(self, input_size, hidden_layers=[128, 64], activations=['relu', 'relu', 'linear'], loss='mse', learning_rate=0.001):
        if len(activations) != len(hidden_layers) + 1:
            raise ValueError("Number of activations must match number of layers + 1 (for output)")

        model = Sequential()
        model.add(Input(shape=(input_size,)))

        for units, activation in zip(hidden_layers, activations[:-1]):
            model.add(Dense(units, activation=activation))

        model.add(Dense(1, activation=activations[-1]))

        model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

        return model
    
    def remember(self, state, action, reward, next_state, done, legal_actions):
        self.memory.append((state, action, reward, next_state, done, legal_actions))
    def act(self, state, legal_actions):
        if random.random() <= self.epsilon:
            return random.choice(legal_actions)
        best_q = -float('inf')
        best_action = None
        for action in legal_actions:
            input_vec = np.concatenate([state, np.array(action)]).reshape(1, -1)
            q_value = self.model.predict(input_vec, verbose=0)[0]
            if q_value > best_q:
                best_q = q_value
                best_action = action

        return best_action
    def train(self):
        if len(self.memory) < self.replay_start_size:
            return

        batch = random.sample(self.memory, self.batch_size)

        x, y = [], []

        for state, action, reward, next_state, done, legal_actions in batch:
            input_vec = np.concatenate([state, np.array(action)]).reshape(1, -1)
            if not done:
                max_future_q = max(self.model.predict(np.concatenate([state, np.array(a)]).reshape(1, -1), verbose=0)[0] for a in legal_actions)
                target_q = reward + self.gamma * max_future_q
            else:
                target_q = reward

            x.append(input_vec)
            y.append(target_q)

        self.model.fit(np.array(x), np.array(y), batch_size=self.batch_size, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
    
    def save_model(self, filepath="tetris_agent.keras"):
        self.model.save(filepath)
        print(f"📦 Model saved to {filepath}")

    def load_model(self, path="tetris_agent.keras"):
        try:
            self.model = load_model(path)
            print(f"✅ Successfully loaded model from {path}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")

def train_tetris_agent(episodes=1000):
    env = LocalTetris()
    agent = TetrisAgent(state_size=env.get_state_size())
    best_score = 0
    scores = []
    for generation in tqdm(range(10)):
        agent.epsilon = 1
        for episode in tqdm(range(episodes)):
            state = env.reset()
            done = False

            while not done:
                next_states_dict = env.get_next_states()
                legal_actions = list(next_states_dict.keys())
                if not next_states_dict:
                    pass
                best_action = agent.act(state, legal_actions)
                next_state, reward, done = env.step(best_action)

                agent.remember(state, best_action, reward, next_state, done, legal_actions)
                state = next_state

            agent.train()
            score = env.get_game_score()
            scores.append(score)

            print(f"[Episode {episode}] Score: {score} | Avg (last 50): {np.mean(scores[-50:]):.2f} | Epsilon: {agent.epsilon:.3f}")

            # Save best model
            if score > best_score:
                best_score = score
                agent.save_model()
                print(f"🎉 New Best Score: {best_score} (Episode {episode})")
        agent.save_model("generation_ai3.keras")
#train_tetris_agent()