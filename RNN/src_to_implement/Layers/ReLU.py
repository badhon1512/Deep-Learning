import numpy as np
from Layers.Base import BaseLayer

class ReLU(BaseLayer):

    def __init__(self):
        super().__init__()


    def forward(self, input_tensor):
        self.x = input_tensor
        #print(np.where(input_tensor < 0, 0, input_tensor))
        return np.where(input_tensor < 0, 0, input_tensor)


    def backward(self, error_tensor):

        self.x = self.x > 0
        # x >1 e or 0
        return error_tensor * self.x
