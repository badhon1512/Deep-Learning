import numpy as np
from Layers.Base import BaseLayer

class TanH(BaseLayer):

    def __init__(self):
        super().__init__()


    def forward(self, input_tensor):
        self.y_hat = np.tanh(input_tensor)
        return self.y_hat


    def backward(self, error_tensor):

        return error_tensor * (1 - pow(self.y_hat, 2))
