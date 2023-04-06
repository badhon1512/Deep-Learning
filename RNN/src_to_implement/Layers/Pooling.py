from Layers.Base import BaseLayer
import numpy as np
import math


class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.max_trackers = []




    def forward(self, input_tensor):

        self.input_shape = input_tensor.shape

        output_y = (self.input_shape[2] - self.pooling_shape[0]) // self.stride_shape[0] + 1
        output_x = (self.input_shape[3] - self.pooling_shape[1]) // self.stride_shape[1] + 1

        output_tensor = np.zeros((self.input_shape[0], self.input_shape[1], output_y, output_x))

        for b in range(self.input_shape[0]):
            for c in range(self.input_shape[1]):

                for y in range(output_y):

                    for x in range(output_x):

                        y_start = y * self.stride_shape[0]
                        y_end = y_start + self.pooling_shape[0]
                        x_start = x * self.stride_shape[1]
                        x_end = x_start + self.pooling_shape[1]

                        data = input_tensor[b, c, y_start:y_end, x_start:x_end]

                        output_tensor[b, c, y, x] = np.max(data)

                        m_y, m_x =np.unravel_index(np.argmax(data, axis=None), data.shape)
                        m_y += y_start
                        m_x += x_start
                        self.max_trackers.append([b, c, m_y, m_x])
        return output_tensor

    def backward(self, error_tensor):

        error_grad = np.zeros(self.input_shape)

        error_data = np.reshape(error_tensor, error_tensor.size)


        for e in range(len(error_data)):

            b, c, y, x = self.max_trackers[e]
            error_grad[b, c, y, x] += error_data[e]


        return error_grad




