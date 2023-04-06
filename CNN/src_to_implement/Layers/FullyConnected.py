from src_to_implement.Layers.Base import BaseLayer
import numpy as np


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.rand(input_size+1, output_size)

    def optimizer(self):
        return self._optimizer

    def optimizer(self, optimizer):
        self._optimizer = optimizer



    def gradient_weights(self):

        return self._weights_error

    def gradient_weights(self, weight_error):

        self._weights_error = weight_error


    def forward(self, input_tensor):

        self.batch_size, self.input_size = input_tensor.shape
        self.x = np.append(input_tensor, np.ones((self.batch_size, 1)), axis=1)
        #y_hat = x*w
        self.y_hat = np.dot(self.x, self.weights)
        return self.y_hat

    def backward(self, error_tensor):

        #En-1 = En * wT
        self.input_error = np.dot(error_tensor, self.weights.T)
        self.gradient_weights = np.dot(self.x.T, error_tensor)


        try:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
        except:
            pass

        return self.input_error[0:, 0:self.input_size]


    def initialize(self, weights_initializer, bias_initializer):

        self.weights = np.append(weights_initializer.initialize((self.input_size,self.output_size), self.input_size, self.output_size),

            bias_initializer.initialize((1,self.output_size), self.output_size, self.output_size),axis=0)
