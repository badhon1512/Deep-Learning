from Layers.Base import BaseLayer
import numpy as np

class Dropout(BaseLayer):

    def __init__(self, probability):
        super().__init__()
        self.probability = probability

    def forward(self, input_tensor):

        if self.testing_phase is True:
            return input_tensor
        else:
            self.keep = np.random.rand(input_tensor.shape[0], input_tensor.shape[1]) < self.probability
            updated_input_tensor = self.keep * input_tensor
            return updated_input_tensor * 1/self.probability

    def backward(self, error_tensor):

        return self.keep * error_tensor / self.probability
