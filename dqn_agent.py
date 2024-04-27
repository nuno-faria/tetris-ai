import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import deque, OrderedDict
import numpy as np
import random

torch.autograd.set_detect_anomaly(True)

class DQNAgent:
    def __init__(self, state_size, mem_size=10000, discount=0.95,
                 epsilon=1, epsilon_min=0, epsilon_stop_episode=500,
                 n_neurons=[100,100,100], activations=['relu', 'relu', 'linear'],
                 loss='mse', optimizer='adam', replay_start_size=None):

        assert len(activations) == len(n_neurons) + 1
        activationsDict = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh, "linear": nn.Identity}

        self.state_size = state_size
        self.memory = deque(maxlen=mem_size)
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / (epsilon_stop_episode)
        self.n_neurons = n_neurons
        self.activations = [activationsDict[x] for x in activations]
        self.loss = loss
        self.optimizer = optimizer
        if not replay_start_size:
            replay_start_size = mem_size / 2
        self.replay_start_size = replay_start_size
        self.model, self.loss, self.optimizer = self._build_model()

    def _build_model(self):
        model = nn.Module()
        layers = []
        n_nodes = [self.state_size] + self.n_neurons + [1]

        assert len(n_nodes) == len(self.activations) + 1

        for layer_cnt in range(len(self.activations)):
            layers.append((f"fc{layer_cnt}", nn.Linear(n_nodes[layer_cnt], n_nodes[layer_cnt + 1])))
            if self.activations[layer_cnt] == "relu":
                layers.append((f"act{layer_cnt}", nn.ReLU()))

        model = nn.Sequential(OrderedDict(layers))
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters())

        return model, criterion, optimizer

    def add_to_memory(self, current_state, next_state, reward, done):
        '''Adds a play to the replay memory buffer'''
        self.memory.append((current_state, next_state, reward, done))

    def random_value(self):
        '''Random score for a certain action'''
        return random.random()

    def predict_value(self, state):
        '''Predicts the score for a certain state'''
        state = torch.tensor(state, dtype=torch.float32)
        return self.model(state).item()

    def act(self, state):
        '''Returns the expected score of a certain state'''
        state = np.reshape(state, [1, self.state_size])
        if random.random() <= self.epsilon:
            return self.random_value()
        else:
            return self.predict_value(state)

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

    def train(self, batch_size=32, epochs=3):
        '''Trains the agent'''
        n = len(self.memory)

        if n >= self.replay_start_size and n >= batch_size:

            batch = random.sample(self.memory, batch_size)

            # Get the expected score for the next states, in batch (better performance)
            next_states = torch.tensor([x[1] for x in batch], dtype=torch.float)
            next_qs = [x[0] for x in self.model.forward(next_states)]

            x = []
            y = []

            # Build xy structure to fit the model in batch (better performance)
            for i, (state, _, reward, done) in enumerate(batch):
                if not done:
                    # Partial Q formula
                    new_q = reward + self.discount * next_qs[i]
                else:
                    new_q = reward

                x.append(state)
                y.append(new_q)
            
            x = torch.tensor(x, dtype=torch.float)
            y = torch.tensor(y, dtype=torch.float)

            # Fit the model to the given values
            dataset = StateScoreDataset(x, y.reshape(-1, 1))
            dataloader = DataLoader(dataset=dataset, batch_size=batch_size)
            for _ in range(epochs):
                for data, label in dataloader:
                    self.optimizer.zero_grad()
                    loss = self.loss(self.model.forward(x), label)
                    loss.backward()
                    self.optimizer.step()

            # Update the exploration variable
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay


class StateScoreDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

        assert len(self.data) == len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.data)