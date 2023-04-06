from Layers.Base import BaseLayer
import numpy as np
from Layers.TanH import TanH
from Layers.Sigmoid import Sigmoid
from Layers.FullyConnected import FullyConnected


class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_state = np.zeros(self.hidden_size)
        self.memorize = False
        self.batch_size = None
        self.TanH = TanH()
        self.Sigmoid = Sigmoid()
        self.f_c_layer = FullyConnected(self.hidden_size + self.input_size, self.hidden_size)
        self.f_c_y_layer = FullyConnected(self.hidden_size , self.output_size)


    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, value):
        self._memorize = value

    @property
    def weights(self):
        return self.f_c_layer.weights

    @weights.setter
    def weights(self, weights):
        self.f_c_layer.weights = weights

    def initialize(self, weights_initializer, bias_initializer):
        self.f_c_layer.initialize(weights_initializer, bias_initializer)
        self.f_c_y_layer.initialize(weights_initializer, bias_initializer)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, weight_error):
        self._gradient_weights = weight_error

    def forward(self, input_tensor):
       self.batch_size = input_tensor.shape[0]
       self.output = np.zeros((self.batch_size, self.hidden_size))
       self.h_s_i_t = np.zeros((self.batch_size, self.input_size + self.hidden_size))

       if self.memorize:
           last_state = self.hidden_state[-1]
       self.hidden_state = np.zeros((self.batch_size + 1, self.hidden_size))

       if  self.memorize:
           self.hidden_state[0] = last_state
       for i in range(input_tensor.shape[0]):
           self.h_s_i_t[i] = np.concatenate((self.hidden_state[i], input_tensor[i]))

           self.hidden_state[i] = self.f_c_layer.forward(self.h_s_i_t[i].reshape(1, -1))
           self.hidden_state[i+1] = self.TanH.forward(self.hidden_state[i].reshape(1, -1))

           self.output[i] = self.hidden_state[i+1]

       #print(self.Sigmoid.forward(self.f_c_y_layer.forward(self.output)).shape)

       self.y_hat = self.Sigmoid.forward(self.f_c_y_layer.forward(self.output))

       return self.y_hat




    def backward(self, error_tensor):

        error_grad = np.zeros((self.batch_size, self.input_size))
        error_tensor = self.f_c_y_layer.backward(self.Sigmoid.backward(error_tensor))
        self.f_c_layer.gradient_weights = np.zeros((self.hidden_size + self.input_size + 1, self.hidden_size))

        error_hidden_input = np.column_stack((self.h_s_i_t, np.ones((self.batch_size))))

        self.gradient_hidden = np.zeros((self.batch_size + 1, self.hidden_size))

        for i in range(self.batch_size):

            self.TanH.activations = self.hidden_state[i + 1, :]

            self.f_c_layer.input_tensor = error_hidden_input[i, :].reshape(1, -1)
            self.gradient_hidden[i, :] = self.gradient_hidden[i + 1, :] + error_tensor[i, :]

            self.gradient_hidden[i, :], error_grad[i, :] = np.hsplit(self.f_c_layer.backward(
                self.TanH.backward(self.gradient_hidden[i, :].reshape(1, -1))), [self.hidden_size])


            self.f_c_layer.gradient_weights += self.f_c_layer.gradient_weights

        try:

            self.f_c_y_layer.weights = self.optimizer.calculate_update(self.f_c_y_layer.weights, self.f_c_y_layer.gradient_weights)
            self.f_c_layer.weights = self.optimizer.calculate_update(self.f_c_layer.weights, self.f_c_layer.gradient_weights)
        except:
            pass

        return error_grad

