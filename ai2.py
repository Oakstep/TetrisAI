import numpy as np
import tensorflow as tf
from keras._tf_keras import keras
from keras._tf_keras.keras import Sequential, Input, Model
from keras._tf_keras.keras.layers import Dense, Flatten, Concatenate, InputLayer
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.optimizers import Adam
from collections import deque
from tetris import LocalTetris
from controller import SimActions
import random
from tqdm import tqdm

class TetrisAgent:
    def __init__(self, state_size, hidden_layers=[128, 64], activations=['relu', 'relu', 'linear'],
                 loss='mse', lr=0.001,
                 memory_size=20000, batch_size=512, replay_start_size=1500, modelFile=None, epsilon=1, episodes=1000):
        self.state_size = state_size
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
            input_size=state_size,
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

        model.compile(loss='mse', optimizer='adam')

        return model
    
    def remember(self, current_state, next_state, reward, done):
        self.memory.append((current_state, next_state, reward, done))

    def train(self, batch_size=512, epochs=3):
        if len(self.memory) < self.replay_start_size:
            return  # Not enough data to train yet

        batch = random.sample(self.memory, batch_size)
        next_states = np.array([x[1] for x in batch])
        next_qs = [x[0] for x in self.model.predict(next_states, verbose=0)]
        # Predict Q-values for current and next states

        x = []
        y = []

        for i, (state, next_state, reward, done) in enumerate(batch):
            if not done:
                new_q  = reward + self.gamma * next_qs[i]
            else:
                new_q  = reward

            x.append(state)
            y.append(new_q)  # wrap in list to match model output shape

        # Train the model
        self.model.fit(np.array(x), np.array(y), batch_size=batch_size, epochs=epochs, verbose=0)

        # Decay exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
    def act(self, state):
        state = np.reshape(state, [1, self.state_size])
        if random.random() <= self.epsilon:
            return random.random()
        else:
            return self.model.predict(state, verbose=0)[0]
        
    def save_model(self, filepath="tetris_agent.keras"):
        self.model.save(filepath)
        print(f"📦 Model saved to {filepath}")

    def load_model(self, path="best_tetris_model.keras"):
        try:
            self.model = load_model(path)
            print(f"✅ Successfully loaded model from {path}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")

    def best_state(self, states):
        '''Returns the best state for a given collection of states'''
        max_value = None
        best_state = None

        if random.random() <= self.epsilon:
            return random.choice(list(states))

        else:
            for state in states:
                value = self.predict_value(np.reshape(state, [1, self.state_size]))
                if not max_value or value > max_value:
                    max_value = value
                    best_state = state

        return best_state
    def predict_value(self, state):
        return self.model.predict(state, verbose=0)[0]


def train_tetris_agent(episodes=1000):
    env = LocalTetris()
    agent = TetrisAgent(state_size=env.get_state_size())
    best_score = 0
    scores = []

    for episode in tqdm(range(episodes)):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Get next possible states
            next_states = {tuple(v):k for k, v in env.get_next_states().items()}
            if not next_states:
                break
            best_state = agent.best_state(next_states.keys())
            best_action = next_states[best_state]
            _, reward, done = env.step(best_action)

            # Remember and train
            agent.remember(state, best_state, reward, done)

            # Update current state
            state = best_state
            total_reward += reward
        agent.train()
        score = env.get_game_score()
        scores.append(score)

        print(f"[Episode {episode}] Score: {score} | Avg (last 50): {np.mean(scores[-50:]):.2f} | Epsilon: {agent.epsilon:.3f}")

        # Save best model
        if score > best_score:
            best_score = score
            agent.save_model("best_tetris_model.keras")
            print(f"🎉 New Best Score: {best_score} (Episode {episode})")
def generational_training(env, generations=10, episodes_per_gen=1000, model_path='tetris_model.keras'):
    agent = TetrisAgent(state_size=env.get_state_size())

    for gen in range(generations):
        print(f"\n🌱 Generation {gen+1} Starting...")

        # Reset epsilon for exploration
        agent.epsilon = 1.0

        if gen > 0:
            agent.load_model(model_path)

        for episode in range(episodes_per_gen):
            state = env.reset()
            done = False
            total_reward = 0

            while not done:
                next_states = {tuple(v):k for k, v in env.get_next_states().items()}
                if not next_states:
                    break
                best_state = agent.best_state(next_states.keys())
                best_action = next_states[best_state]
                _, reward, done = env.step(best_action)

                # Remember and train
                agent.remember(state, best_state, reward, done)

                # Update current state
                state = best_state
                total_reward += reward

            print(f"Episode {episode + 1} Score: {env.score}")
            agent.train()
        agent.save_model(model_path)
        print(f"💾 Saved model after Generation {gen+1}")
env = LocalTetris()
#train_tetris_agent()
#generational_training(env=env)