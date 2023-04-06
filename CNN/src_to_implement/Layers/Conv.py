import copy
import math

from src_to_implement.Layers.Base import BaseLayer
import numpy as np
from scipy import signal


class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.is_2d = False
        if len(self.convolution_shape) == 3:
            self.is_2d = True
            self.weights = np.random.rand(self.num_kernels , self.convolution_shape[0],self.convolution_shape[1],self.convolution_shape[2])

        else:
            self.weights = np.random.rand(self.num_kernels , self.convolution_shape[0],self.convolution_shape[1])

        self.bias = np.random.random(self.num_kernels)
        print(self.convolution_shape)

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

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weights.shape, np.prod(self.convolution_shape),
                                                           np.prod(np.prod(self.convolution_shape[1:]) * self.num_kernels))
        self.bias = bias_initializer.initialize(self.bias.shape, np.prod(self.convolution_shape),
                                                        np.prod(np.prod(self.convolution_shape[1:]) * self.num_kernels))

    def forward(self, input_tensor):

        input_shape = input_tensor.shape
        self.input_tensor = input_tensor

        output_y = math.ceil((input_shape[2]) / self.stride_shape[0])

        if self.is_2d:
            output_x = math.ceil((input_shape[3]) / self.stride_shape[1])
            self.output = np.zeros((input_shape[0], self.num_kernels, output_y, output_x))

        else:
            self.output = np.zeros((input_shape[0], self.num_kernels, output_y))

        for b in range(self.output.shape[0]):
            for k in range(self.num_kernels):
                for c in range(input_shape[1]):
                    if self.is_2d:


                        self.output[b, k] += signal.correlate(input_tensor[b, c], self.weights[k, c], 'same')[::self.stride_shape[0], :: self.stride_shape[1]]

                    else:
                        self.output[b, k] += signal.correlate(input_tensor[b, c], self.weights[k, c], 'same')[::self.stride_shape[0]]

                self.output[b, k] += self.bias[k]

        return self.output



    def backward(self, error_tensor):


       error_grad = np.zeros(self.input_tensor.shape)
       error_shape = error_tensor.shape
       if self.is_2d:
           update_error_tensor = np.zeros((error_shape[0], error_shape[1], self.input_tensor.shape[2], self.input_tensor.shape[3]))
       else:
           update_error_tensor = np.zeros((error_shape[0], error_shape[1], self.input_tensor.shape[2]))

       for b in range(error_shape[0]):
           for k in range(error_shape[1]):
               if self.is_2d:
                    update_error_tensor[b, k, ::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor[b, k,:,:]
               else:
                   update_error_tensor[b, k, ::self.stride_shape[0]] = error_tensor[b, k]

       for b in range(error_grad.shape[0]):
           for c in range(error_grad.shape[1]):
               for k in range(self.num_kernels):
                   if self.is_2d:
                       error_grad[b, c] += signal.convolve(update_error_tensor[b, k], self.weights[k, c], 'same')
                   else:
                       error_grad[b, c] += signal.convolve(update_error_tensor[b, k], self.weights[k, c], 'same')
       self.gradient_weights = np.zeros(self.weights.shape)

       if self.is_2d:
           update_input_tensor = np.pad(self.input_tensor, ((0, 0), (0, 0), (int(self.convolution_shape[1] / 2), int((self.convolution_shape[1] ) / 2)),
                                            (int(self.convolution_shape[2] / 2), int((self.convolution_shape[2]-1) / 2))))
       else:
           update_input_tensor = np.pad(self.input_tensor,
                            ((0, 0), (0, 0), (int(self.convolution_shape[1] / 2), int((self.convolution_shape[1] - 1) / 2))))

       for b in range(error_grad.shape[0]):
           for k in range(self.num_kernels):
               for c in range(error_grad.shape[1]):
                   if self.is_2d:
                       print(signal.correlate(update_input_tensor[b,c],update_error_tensor[b, k],  'valid').shape)

                       self.gradient_weights[k, c] += signal.correlate(update_input_tensor[b,c],update_error_tensor[b, k],  'valid')

                   else:
                       self.gradient_weights[k, c] += signal.correlate(update_input_tensor[b,c],update_error_tensor[b, k],  'valid')




       if self.is_2d:
            self.gradient_bias = np.sum(error_tensor, axis=(0, 2, 3))
       else:
            self.gradient_bias = np.sum(error_tensor, axis=(0, 2))

       try:
           self.weights = self._optimizer_weights.calculate_update(self.weights, self.gradient_weights)

       except:
           pass

       try:
           self.bias = self._optimizer_bias.calculate_update(self.bias, self.gradient_bias)

       except:
           pass


       return error_grad

a=[1,2,3]
b=[1,2,3]

print(np.correlate(a,b,'valid'))