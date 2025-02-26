import torch
import random
from collections import namedtuple

""" From "actin-control" repo (Shriram Chennakesavalu, Sreekanth Manikandan, Frank Hu, Grant Rotskoff, 2024)
'Adaptive nonequilibrium design of actin-based metamaterials: fundamental and practical limits of control'
"""

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = int(0)
        self.transition = namedtuple("Transition", 
                                     ("state", "action", "reward", "next_state", "done"))

    def push(self, state, action, reward, next_state, done):
        to_add = [state, action, reward, next_state, done]
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = self.transition(*to_add)
        self.position = int((self.position + 1) % self.capacity)

    def get_element_from_buffer(self, element_num):
        return self.buffer[element_num]    

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return (len(self.buffer))

    def load_buffer(self, filename):
        self.buffer = np.load(filename, allow_pickle=True).tolist()
        self.position = len(self.buffer)