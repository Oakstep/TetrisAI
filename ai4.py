
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

class TetrisAgent:
    def __init__(self, state_size, modelFile = None, epsilon = 1.0):
        self.state_size = state_size
        self.memory = deque(maxlen=30000)
        self.discount = 0.95
        self.epsilon = epsilon
        self.epsilon_min = 0.001 
        self.epsilon_end_episode = 1500
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.epsilon_end_episode

        self.batch_size = 512
        self.replay_start = 3000
        self.epochs = 3
        if modelFile is not None:
            self.model = load_model(modelFile)
        else:
            self.model = self.build_model()
    def build_model(self):
        model = Sequential([
                Dense(64, input_dim=self.state_size, activation='relu'),
                Dense(64, activation='relu'),
                Dense(32, activation='relu'),
                Dense(1, activation='linear')
        ])

        model.compile(loss='mse', optimizer='adam')
        return model
    def add_to_memory(self, current_state, next_state, reward, done):
        self.memory.append((current_state, next_state, reward, done))
    def act(self, states):
        max_value = -float('inf')
        best = None

        if random.random() <= self.epsilon:
            return random.choice(list(states))
        else:
            for state in states:
                value = self.model.predict(np.reshape(state, [1, self.state_size]), verbose=0)
                if value > max_value:
                    max_value = value
                    best = state
        return best
    def train(self):
        if len(self.memory) > self.replay_start:
            batch = random.sample(self.memory, self.batch_size)
            next_states = np.array([x[1] for x in batch])
            next_q = np.array([x[0] for x in self.model.predict(next_states, verbose=0)])
            x = []
            y = []
            for i, (state, _, reward, done) in enumerate(batch):
                if not done:
                    new_q = reward + self.discount*next_q[i]
                else:
                    new_q = reward
                x.append(state)
                y.append(new_q)
            self.model.fit(np.array(x), np.array(y), batch_size=self.batch_size, epochs=self.epochs, verbose=0)
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay
    def save_model(self, name='tetris_model.keras'):
        self.model.save(name)