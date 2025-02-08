import numpy as np


class CrossEntropyLoss:

    def __init__(self):
        pass

    def forward(self, prediction_tensor, label_tensor):

        self.y = label_tensor
        self.y_hat = prediction_tensor
        #print(np.finfo(float).eps)
        return (np.where(self.y == 1, -np.log(self.y_hat+np.finfo(float).eps), 0)).sum()

    def backward(self, label_tensor):

        return - label_tensor / (self.y_hat + np.finfo(float).eps)

