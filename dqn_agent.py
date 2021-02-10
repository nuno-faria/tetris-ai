
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras.models import Sequential, load_model, Model
# from keras.layers import Dense
from keras import layers, optimizers
from collections import deque
import numpy as np
import random
import pickle
from statistics import median

# Deep Q Learning Agent + Maximin
#
# This version only provides only value per input,
# that indicates the score expected in that state.
# This is because the algorithm will try to find the
# best final state for the combinations of possible states,
# in constrast to the traditional way of finding the best
# action for a particular state.
class DQNAgent:

    '''Deep Q Learning Agent + Maximin

    Args:
        state_size (int): Size of the input domain
        mem_size (int): Size of the replay buffer
        discount (float): How important is the future rewards compared to the immediate ones [0,1]
        epsilon (float): Exploration (probability of random values given) value at the start
        epsilon_min (float): At what epsilon value the agent stops decrementing it
        epsilon_stop_episode (int): At what episode the agent stops decreasing the exploration variable
        loss (obj): Loss function
        optimizer (obj): Otimizer used
        replay_start_size: Minimum size needed to train
    '''

    def __init__(self, state_size, mem_size=10000, discount=0.95,
                 epsilon=0.1, epsilon_min=0.1, epsilon_stop_episode=1000,
                 nb_channels_small_filter=6, nb_channels_big_filter=8, nb_dense_neurons=[12,4,4],
                 loss='mse', optimizer='adam', replay_start_size=None):
        self.state_size = state_size
        self.memory = deque(maxlen=mem_size)
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / (epsilon_stop_episode)
        self.nb_channels_small_filter = nb_channels_small_filter
        self.nb_channels_big_filter   = nb_channels_big_filter
        self.nb_dense_neurons         = nb_dense_neurons
        self.loss = loss
        # self.optimizer = optimizer
        optim_params  = optimizer.split('-')
        self.adam_lr  = 10**-int(optim_params[-2]) if len(optim_params)>2 else 0.001
        self.adam_eps = 10**-int(optim_params[-1]) if len(optim_params)>2 else 1e-7
        if not replay_start_size:
            replay_start_size = mem_size / 2
        self.replay_start_size = replay_start_size
        self.model = self._build_model()


    def _build_model(self):
        '''Builds a Keras deep neural network model'''
        input_board  = layers.Input(shape=(6, 6, 1))
        all_nn_after_conv2d = []
        for kernel_size, nb_chan in [ (x, self.nb_channels_small_filter) for x in [(1,2), (2,1)] ]:
            pool_size = (1,5) if kernel_size[0]<kernel_size[1] else (5,1)
            nn10 = layers.ReLU()(input_board)
            nn20 = layers.Conv2D(nb_chan, kernel_size, activation='relu')(nn10)
            nn30 = layers.AveragePooling2D(pool_size=pool_size)(nn20)
            nn40 = layers.GlobalMaxPool2D()(nn30)
            all_nn_after_conv2d.append(nn40)
        small_filter_nn = layers.Dense(self.nb_dense_neurons[0], activation='relu')(layers.concatenate(all_nn_after_conv2d))
        all_nn_after_conv2d = []
        for kernel_size, nb_chan in [ (x, self.nb_channels_big_filter) for x in [(2,3), (3,2), (3,3)] ]:
            nn15 = layers.ReLU()(input_board)
            nn25 = layers.Conv2D(nb_chan, kernel_size, activation='relu')(nn15)
            nn35 = layers.GlobalAveragePooling2D()(nn25)
            all_nn_after_conv2d.append(nn35)
        big_filter_nn = layers.Dense(self.nb_dense_neurons[1], activation='relu')(layers.concatenate(all_nn_after_conv2d))

        input_scalarA  = layers.Input(shape=(1, ))
        input_scalarB  = layers.Input(shape=(1, ))
        new_dense = layers.concatenate([small_filter_nn, big_filter_nn, input_scalarA, input_scalarB])
        # new_dense = layers.concatenate([small_filter_nn, big_filter_nn])
        for dense in self.nb_dense_neurons[2:]:
            previous_dense = new_dense
            new_dense = layers.Dense(dense)(previous_dense)
        end_nn = layers.Dense(1)(new_dense)
        model = Model(inputs=[input_scalarA, input_scalarB, input_board], outputs=end_nn)
        # model = Model(inputs=input_board, outputs=end_nn)
        model.compile(loss=self.loss, optimizer=optimizers.Adam(learning_rate=self.adam_lr, epsilon=self.adam_eps))
        model.summary()
        
        return model


    def add_to_memory(self, current_state, next_state, reward, done):
        '''Adds a play to the replay memory buffer'''
        self.memory.append((current_state, next_state, reward, done))


    def random_value(self):
        '''Random score for a certain action'''
        return random.random()


    def predict_value(self, state):
        '''Predicts the score for a certain state'''
        return self.model.predict(state)[0]


    def act(self, state):
        '''Returns the expected score of a certain state'''
        state = np.reshape(state, [1, self.state_size])
        if random.random() <= self.epsilon:
            return self.random_value()
        else:
            return self.predict_value(state)


    def find_best_state(self, states, allow_random=True):
        '''Returns the best state/action for a given dict of states/action'''
        if len(states) == 0:
            return None, None
        max_value = None
        best_action, best_state = None, None

        if allow_random and random.random() <= self.epsilon:
            return random.choice(list(states.items()))

        all_actions_states = list(states.items())
        all_states = [ [s[1][i] for s in all_actions_states] for i in range(3) ]

        all_values = self.model.predict(all_states, workers=6, use_multiprocessing=True)
        best_i = np.argmax(all_values)
        best_action, best_state = all_actions_states[best_i]
        return best_action, best_state            


    def train(self, batch_size=32, epochs=3, force_fail=10):
        '''Trains the agent'''
        n = len(self.memory)
    
        if n >= self.replay_start_size and n >= batch_size:

            memory_fail = [ x for x in self.memory if x[3] ]
            # memory_fail = memory_fail[-batch_size:] # taking recent failures
            batch1 = random.sample(memory_fail, min(batch_size*force_fail//100, len(memory_fail)))
            batch2 = random.sample(self.memory, batch_size-len(batch1))
            batch = batch2 + batch1

            # Get the expected score for the next states, in batch (better performance)
            next_states = [ [s[1][i] for s in batch] for i in range(3) ]
            next_qs = self.model.predict(next_states, workers=6, use_multiprocessing=True)

            cur_states  = [ [s[0][i] for s in batch] for i in range(3) ]
            y = []
            # Build xy structure to fit the model in batch (better performance)
            for i, (_, _, reward, done) in enumerate(batch):
                # Partial Q formula
                new_q = reward + ((self.discount * next_qs[i]) if not done else 0)
                y.append(new_q)
            # Fit the model to the given values
            history = self.model.fit(cur_states, np.array(y), batch_size=batch_size, epochs=epochs, verbose=0, workers=6, use_multiprocessing=True)
            # Evalute model on failing states
            fail_qs = self.model.predict([next_states[i][-len(batch1):] for i in range(3)], workers=6, use_multiprocessing=True)[0]
            # print('Failure results: ', max(fail_qs), median(fail_qs), fail_qs)
            # Update the exploration variable
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay
            return (history.history['loss'][-1], median(fail_qs))
        return (None, None)


    def save(self, filename):
        self.model.save(filename + '.h5')
        with open(filename + '.pickle', 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
