import numpy as np
from src_to_implement.Layers.Base import BaseLayer

class Flatten(BaseLayer):

    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.shape = input_tensor.shape
        return np.reshape(input_tensor, (self.shape[0], input_tensor.size // self.shape[0]))

    def backward(self, error_tensor):
        return np.reshape(error_tensor, self.shape)
