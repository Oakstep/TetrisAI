import cv2
import numpy as np
import tensorflow as tf
from keras._tf_keras import keras
from keras._tf_keras.keras import Sequential, Input, Model
from keras._tf_keras.keras.layers import Dense, Flatten, Concatenate
from collections import deque
from tetris import Tetris
from controller import SimActions

class AIPart:
    def __init__(self):
        self.model = self.create_model()  # Build the neural network model
        self.memory = deque(maxlen=4000)  # Replay memory for storing experiences
        self.gamma = 0.95  # Discount factor for future rewards
        self.epsilon = 1.0  # Exploration-exploitation balance
        self.epsilon_min = 0.01  # Minimum exploration probability
        self.epsilon_decay = 0.995  # Decay rate for epsilon
        #self.simActions = simActions

    def create_model(self):
        grid_input = Input(shape=(20, 10), name="grid_input")
        flat_grid = Flatten()(grid_input)  # Flatten the board into a 1D array

        current_piece_input = Input(shape=(7,), name="current_piece_input")  # Assume 7 possible pieces

        held_piece_input = Input(shape=(7,), name="held_piece_input")  # Assume 7 possible pieces

        next_pieces_input = Input(shape=(5, 7), name="next_pieces_input")  # 5 pieces, 7 types each
        flat_next_pieces = Flatten()(next_pieces_input)  # Flatten the 5x7 array
        combined = Concatenate()([flat_grid, current_piece_input, held_piece_input, flat_next_pieces])

        dense1 = Dense(128, activation="relu")(combined)
        dense2 = Dense(128, activation="relu")(dense1)
        dense3 = Dense(128, activation="relu")(dense2)

        output = Dense(8, activation="linear", name="q_values")(dense3) # what the fuck is the output? A location? so like (row, col, rotation)?

        model = Model(
            inputs=[grid_input, current_piece_input, held_piece_input, next_pieces_input],
            outputs=output,
        )
        model.compile(optimizer="adam", loss="mse")
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, grid, current_piece, held_piece, next_pieces):
        inputs = {
            "grid_input": grid.reshape(1, 20, 10),
            "current_piece_input": current_piece.reshape(1, 7),
            "held_piece_input": held_piece.reshape(1, 7),
            "next_pieces_input": next_pieces.reshape(1, 5, 7),
        }
        if np.random.rand() <= self.epsilon:
            return self.generate_random_target(grid, current_piece) # Exploration: Random end position
        else: # Exploitation: Choose the best move
            q_values = self.model.predict(inputs)
            target = self.select_target_position(grid, q_values[0])
        rotation_type = self.classify_rotation(grid, target, current_piece)
        return {"target": target, "rotation_type": rotation_type, "final_rotation": target["rotation"]}
    def generate_random_target(self, grid, current_piece):
        # For simplicity, generate random column and rotation
        target_column = np.random.randint(0, grid.shape[1])
        rotation = np.random.choice(["clockwise", "counterclockwise", "flip"])
        return {"column": target_column, "rotation": rotation}
    def select_target_position(self, grid, q_values):
        best_action_index = np.argmax(q_values)
        # Map the best action index to a column and rotation strategy
        target_column = best_action_index % grid.shape[1]
        rotation = ["clockwise", "counterclockwise", "flip"][best_action_index // grid.shape[1]]
        return {"column": target_column, "rotation": rotation}
    
    def calculate_reward(self, back_to_back):
        reward = 0
        lines_cleared = 0
        #tspin = False
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

    def load(self, name):
        self.model = keras.models.load_model(name)

    def save(self, name):
        self.model.save(name)
    def train(self, env: Tetris, episodes, batch_size):
        pass


def train_ai(env, agent: AIPart, episodes, batch_size):
    for episode in range(episodes):
        print(f"Starting Episode {episode + 1}/{episodes}")
        state = env.reset()
        state = preprocess_state(state)

        total_reward = 0
        steps = 0

        while True:
            # Select action
            action = agent.act(**state)

            # Perform action and get feedback
            next_state, reward, done = env.step(action)

            # Store the experience
            agent.remember(state, action, reward, preprocess_state(next_state), done)

            # Update the state
            state = preprocess_state(next_state)
            total_reward += reward
            steps += 1

            if done:
                print(f"Episode {episode + 1}/{episodes}: Total Reward = {total_reward}, Steps = {steps}")
                break

        # Train the agent after each episode
        agent.replay(batch_size)

        # Reduce exploration rate (epsilon)
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
actions = SimActions()
#agent = AIPart(actions)
env = Tetris()
actions.stream(env)
cv2.waitKey(0)
cv2.destroyAllWindows()
#train_ai(env, agent, episodes=1000, batch_size=32)
