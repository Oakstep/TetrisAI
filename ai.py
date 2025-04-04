from pickle import NONE
from tkinter import NO
import numpy as np
import tensorflow as tf
from keras._tf_keras import keras
from keras._tf_keras.keras import Sequential, Input, Model
from keras._tf_keras.keras.layers import Dense, Flatten, Concatenate
from keras._tf_keras.keras.models import load_model
from collections import deque
from tetris import Tetris, LocalTetris
from controller import SimActions
import random
from tqdm import tqdm

class AIPart:
    def __init__(self, load=True):
        if load is True:
            self.model = load_model("tetris_ai_model.keras")
            self.epsilon = 0.0
        else:
            self.model = self.create_model()
            self.epsilon = 1.0
        self.memory = deque(maxlen=20000)
        self.replay_start_size = 1000
        self.gamma = 0.99
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.max_steps = 2000

    def create_model(self):
        grid_input = Input(shape=(20, 10), name="grid")
        flat_grid = Flatten()(grid_input)

        current_piece_input = Input(shape=(7,), name="current_piece")
        held_piece_input = Input(shape=(7,), name="held_piece")
        next_pieces_input = Input(shape=(5, 7), name="queue")
        flat_next_pieces = Flatten()(next_pieces_input)

        can_hold_input = Input(shape=(1,), name="can_hold")

        combined = Concatenate()([flat_grid, current_piece_input, held_piece_input, flat_next_pieces, can_hold_input])
        dense1 = Dense(128, activation="relu")(combined)
        dense2 = Dense(128, activation="relu")(dense1)
        dense3 = Dense(128, activation="relu")(dense2)

        output = Dense(41, activation="linear", name="q_values")(dense3)
        model = Model(
            inputs=[grid_input, current_piece_input, held_piece_input, next_pieces_input, can_hold_input],
            outputs=output,
        )
        model.compile(optimizer="adam", loss="mse")
        return model
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, simulator_state, legal_actions):
        if not legal_actions: 
            return None
        
        inputs = {
            "grid": simulator_state["grid"],
            "current_piece": simulator_state["current_piece"],
            "held_piece": simulator_state["held_piece"],
            "queue": simulator_state["queue"],
            "can_hold": simulator_state["can_hold"]
        }

        if np.random.rand() <= self.epsilon:
            return random.choice(legal_actions) 
        else:
            q_values = self.model.predict(inputs, verbose=0)[0]
            best_action = max(legal_actions, key=lambda action: q_values[action["index"]])
            return best_action
        
    def train(self, simulator:LocalTetris, episodes):
        scores = []
        best_score = -float("inf")
        for episode in tqdm(range(episodes)):
            simulator.reset()
            state = simulator.get_state()
            done = False
            steps = 0
            while not done and (steps < self.max_steps):
                next_states = simulator.get_next_states()
                best_action_key = max(next_states, key=lambda k: next_states[k][1])
                best_action = dict(best_action_key)
                best_state, reward, done = simulator.step(best_action)
                self.remember(state, best_action, reward, best_state, done)
                state = best_state
                steps+=1
            scores.append(simulator.score)
            if(len(self.memory) >= self.replay_start_size):
                self.replay()
            score = simulator.score
            if score > best_score:
                print(f"Saving a new best model (score={score}, episode={episode})")
                best_score = score
                self.model.save("tetris_ai_model.keras")
            print(f"Episode {episode + 1}/{episodes} completed. Epsilon: {self.epsilon:.4f}")
    def replay(self, batch_size=256):
        if len(self.memory) < batch_size:
            return  # Do nothing if there isn't enough data for a batch
        
        minibatch = random.sample(self.memory, batch_size)  # Sample experiences from memory
        
        for state, action, reward, next_state, done in minibatch:
            # Prepare inputs for prediction from state and next_state
            state_inputs = {
                "grid": state["grid"],
                "current_piece": state["current_piece"],
                "held_piece": state["held_piece"],
                "queue": state["queue"],
                "can_hold": state["can_hold"]
            }
            next_state_inputs = {
                "grid": next_state["grid"],
                "current_piece": next_state["current_piece"],
                "held_piece": next_state["held_piece"],
                "queue": next_state["queue"],
                "can_hold": next_state["can_hold"]
            }
            current_q_values = self.model.predict(state_inputs, verbose=0)[0]
            next_q_values = self.model.predict(next_state_inputs, verbose=0)[0]
            
            # Target computation based on reward and next state
            target = reward
            if not done:
                target += self.gamma * np.max(next_q_values)  # Add discounted max Q-value of next state

            # Create updated Q-values for the current state
            target_q_values = current_q_values.copy()
            if action["index"] < len(target_q_values):  # Update only valid action index
                target_q_values[action["index"]] = target


            # Train the model on this adjusted target
            self.model.fit(state_inputs, np.expand_dims(target_q_values, axis=0), epochs=1, verbose=0)

        # Decay epsilon for better exploration-exploitation balance
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def test(self):
        test_inputs = {
            "grid": np.zeros((1, 20, 10)),
            "current_piece": np.zeros((1, 7)),
            "held_piece": np.zeros((1, 7)),
            "queue": np.zeros((1, 5, 7)),
            "can_hold": np.array([[1]])
        }
        try:
            q_values = self.model.predict(test_inputs, verbose=0)
            print("Q-values:", q_values)
        except AttributeError as e:
            print("Error:", e)

def temp():
    simulator = LocalTetris()
    simulator.reset()
    ai = AIPart(load=False)
    ai.train(simulator, episodes=2000)
    ai.model.save("tetris_ai_model.keras")
#temp()