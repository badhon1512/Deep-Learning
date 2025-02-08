import numpy as np
from src_to_implement.Layers.Base import BaseLayer

class SoftMax(BaseLayer):

    def __init__(self):
        super().__init__()


    def forward(self, input_tensor):

        self.x = input_tensor
        # to prevent the large value
        #y_hat = exp(xk)/sum(xj)

        updated_x = np.subtract(input_tensor, np.max(input_tensor, 1 , keepdims=True))
        self.y_hat = np.exp(updated_x)/np.sum(np.exp(updated_x), 1, keepdims=True)

        return self.y_hat

    def backward(self, error_tensor):

        #E = Y_hat * ( en - sumof(ei*y_hat))
        ey_sum = np.sum((error_tensor * self.y_hat),axis=1, keepdims= True)

        return self.y_hat * (error_tensor-ey_sum)

