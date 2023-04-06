import numpy as np
import numpy.testing as npt

class Optimizer:

    def __init__(self):
        self.regularizer = None


    def add_regularizer(self,regularizer):
        self.regularizer = regularizer


class Sgd(Optimizer):

    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer is not None:
            weight_tensor -= self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)

        return weight_tensor - self.learning_rate*gradient_tensor


class SgdWithMomentum(Optimizer):

    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer is not None:
            weight_tensor -= self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)

        self.v = self.momentum_rate * self.v - self.learning_rate * gradient_tensor

        return weight_tensor + self.v



class Adam(Optimizer):

    def __init__(self, learning_rate, mu, rho):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.k = 1
        self.v = 0
        self.r = 0

    def calculate_update(self, weight_tensor, gradient_tensor):

        if self.regularizer is not None:
            weight_tensor -= self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        self.v = self.mu * self.v + (1-self.mu) * gradient_tensor
        self.r = self.rho * self.r + (1-self.rho) * pow(gradient_tensor , 2)

        v_k = self.v / (1 - pow(self.mu, self.k))
        r_k = self.r / (1 - pow(self.rho, self.k))

        self.k += 1


        return weight_tensor - self.learning_rate*v_k/(np.sqrt(r_k)+np.finfo(float).eps)


# gfg = npt.assert_almost_equal(0.0, np.array([0.0001]))
# print(gfg)
