import numpy as np
import tensorflow as tf
from keras._tf_keras import keras
from keras._tf_keras.keras import Sequential, Input, Model
from keras._tf_keras.keras.layers import Dense, Flatten, Concatenate
from collections import deque
from tetris import Tetris, LocalTetris
from controller import SimActions
import random

class AIPart:
    def __init__(self):
        self.model = self.create_model()
        self.memory = deque(maxlen=4000)  # Replay memory for storing experiences
        self.gamma = 0.95  # Discount factor for future rewards
        self.epsilon = 1.0  # Exploration-exploitation balance
        self.epsilon_min = 0.01  # Minimum exploration probability
        self.epsilon_decay = 0.995  # Decay rate for epsilon

    def create_model(self):
        # Define the neural network architecture
        grid_input = Input(shape=(20, 10), name="grid_input")
        flat_grid = Flatten()(grid_input)

        current_piece_input = Input(shape=(7,), name="current_piece_input")
        held_piece_input = Input(shape=(7,), name="held_piece_input")
        next_pieces_input = Input(shape=(5, 7), name="next_pieces_input")
        flat_next_pieces = Flatten()(next_pieces_input)

        combined = Concatenate()([flat_grid, current_piece_input, held_piece_input, flat_next_pieces])
        dense1 = Dense(128, activation="relu")(combined)
        dense2 = Dense(128, activation="relu")(dense1)
        dense3 = Dense(128, activation="relu")(dense2)

        output = Dense(32, activation="linear", name="q_values")(dense3)  # Adjusted for larger action space
        model = Model(
            inputs=[grid_input, current_piece_input, held_piece_input, next_pieces_input],
            outputs=output,
        )
        model.compile(optimizer="adam", loss="mse")
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, simulator_state, legal_actions):
    # Prepare model inputs from the simulator state
        inputs = {
            "grid_input": simulator_state["grid"].reshape(1, 20, 10),
            "current_piece_input": np.zeros((1, 7)),  # Placeholder for current piece one-hot encoding
            "held_piece_input": np.zeros((1, 7)),  # Placeholder for held piece one-hot encoding
            "next_pieces_input": np.zeros((1, 5, 7)),  # Placeholder for next pieces one-hot encoding
            "can_hold_input": simulator_state["can_hold"]
        }

        if np.random.rand() <= self.epsilon:
            return random.choice(legal_actions)  # Exploration
        else:
            q_values = self.model.predict(inputs, verbose=0)[0]
            # Select the legal action with the highest Q-value
            best_action = max(legal_actions, key=lambda action: q_values[action["index"]])
            return best_action

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_q_vals = self.model.predict(next_state, verbose=0)
                target += self.gamma * np.amax(next_q_vals)
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action["index"]] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, simulator: LocalTetris, episodes, batch_size):
        for episode in range(episodes):
            state = simulator.reset()
            done = False
            while not done:
                legal_actions = simulator.get_legal_actions()
                action = self.act(state, legal_actions)
                next_state, reward, done = simulator.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                self.replay(batch_size)
            print(f"Episode {episode + 1}/{episodes} completed. Epsilon: {self.epsilon:.4f}")

    def calculate_reward(self, back_to_back):
        reward = 0
        lines_cleared = 0
        action = None
        for row in self.grid:
            lines_cleared += 1 if all(row) else 0 
        if lines_cleared == 1:
            action = "single"
            reward += 0
        elif lines_cleared == 2:
            action = "double"
            reward += 1
        elif lines_cleared == 3:
            action = "triple"
            reward += 2
        elif lines_cleared == 4:  # Tetris
            action = "tetris"
            reward += 4
        if(back_to_back == action):
            reward += 1
        #reward -= holes * 0.5
        #reward -= height * 0.2
        return reward

simulator = LocalTetris()
ai = AIPart()
ai.train(simulator, episodes=1000, batch_size=32)
ai.model.save("tetris_ai_model.h5")