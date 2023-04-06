import numpy as np
from Layers.Base import BaseLayer

class Sigmoid(BaseLayer):

    def __init__(self):
        super().__init__()


    def forward(self, input_tensor):

        self.y_hat = 1/(1+np.exp(-input_tensor))

        return self.y_hat


    def backward(self, error_tensor):




        return error_tensor * self.y_hat * (1-self.y_hat)
