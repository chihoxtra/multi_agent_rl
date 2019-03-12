from collections import deque
import random
from utilities import transpose_list


class ReplayBuffer:
    def __init__(self,size):
        self.size = size
        self.memory = deque(maxlen=self.size)

    def add(self, data):
        """add into the buffer"""

        self.memory.append(data)

    def sample(self, batchsize):
        """sample from the buffer"""
        samples = random.sample(self.memory, batchsize)

        print(len(transpose_list(samples)))

        # transpose list of list
        return transpose_list(samples)

    def __len__(self):
        return len(self.deque)
