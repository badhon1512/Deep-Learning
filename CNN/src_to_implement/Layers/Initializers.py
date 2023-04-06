import numpy as np

class Constant:

    def __init__(self, constant_value):
        self.constant_value = constant_value

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.full(weights_shape, self.constant_value)


class UniformRandom:

    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):

        return np.random.uniform(0,1,  weights_shape)


class Xavier:

    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):

        return np.random.normal(0, np.sqrt(2/(fan_in+fan_out)), weights_shape)


class He:

    def __init__(self):
        pass
    def initialize(self, weights_shape, fan_in, fan_out):


        return np.random.normal(0, np.sqrt(2/fan_in), weights_shape)


