from Layers.Base import BaseLayer
import numpy as np
import Layers.Helpers as hp
import copy


class BatchNormalization(BaseLayer):

    def __init__(self, channels):
        super().__init__()
        self.trainable = True
        self.channels = channels
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)
        self.decay = 0.8
        self.mean = None
        self.variance = None

    @property
    def optimizer(self):
        return self._optimizer_weights, self._optimizer_bias
    @optimizer.setter
    def optimizer(self, value):
        self._optimizer_weights = value
        self._optimizer_bias = copy.deepcopy(value)


    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, weight_error):
        self._gradient_weights = weight_error

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, bias_error):
        self._gradient_bias = bias_error

    def initialize(self):
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        if len(self.input_tensor.shape) == 4:
            self.r_i_t = self.reformat(self.input_tensor)
        else:
            self.r_i_t = self.input_tensor

        if self.testing_phase is True:
            self.mean =  self.decay * self.mean + (1-self.decay) * self.batch_mean
            self.variance = self.decay * self.variance + (1-self.decay) * self.batch_variance
            self.x_hat = (self.r_i_t - self.mean) / np.sqrt(self.variance - np.finfo(float).eps)

        else:
            self.batch_mean = np.mean(self.r_i_t, axis=0)
            self.batch_variance = np.var(self.r_i_t, axis=0)
            if self.mean is None:
                self.mean = self.batch_mean
                self.variance = self.batch_variance

            self.x_hat = (self.r_i_t - self.batch_mean) / np.sqrt(self.batch_variance - np.finfo(float).eps)

        self.y_hat = self.x_hat * self.weights + self.bias

        if len(self.input_tensor.shape) == 4:
            self.y_hat =  self.reformat( self.y_hat)
        else:
            self.y_hat =  self.y_hat

        return self.y_hat

    def backward(self, error_tensor):
        if len(self.input_tensor.shape) == 4:
            r_e_t = self.reformat(error_tensor)
        else:
            r_e_t = error_tensor

        error_grad = hp.compute_bn_gradients(r_e_t,self.r_i_t,self.weights,self.batch_mean,self.batch_variance, np.finfo(float).eps)

        self.gradient_weights = np.sum(r_e_t * self.x_hat , axis=0)
        self.gradient_bias = np.sum(r_e_t, axis=0)

        try:
            self.weights = self._optimizer_weights.calculate_update(self.weights, self.gradient_weights)
        except:
            pass

        try:
            self.bias = self._optimizer_bias.calculate_update(self.bias, self.gradient_bias)

        except:
            pass
        # 4 --> 2
        # 2 --> 4
        if len(self.input_tensor.shape) == 4:
            error_grad = self.reformat(error_grad)
        else:
            error_grad = error_grad
        return error_grad


    def reformat(self,tensor):

        if len(tensor.shape) == 4:


            return np.concatenate(np.reshape(tensor, (tensor.shape[0],  tensor.shape[1], tensor.shape[2]* tensor.shape[3])),axis=1).T
        elif len(tensor.shape) == 2:

            return np.reshape(np.transpose(np.reshape(tensor, (self.input_tensor.shape[0], self.input_tensor.shape[2] * self.input_tensor.shape[3], self.input_tensor.shape[1]))
                                , (0, 2, 1)), (self.input_tensor.shape[0],self.input_tensor.shape[1], self.input_tensor.shape[2],self.input_tensor.shape[3]))







