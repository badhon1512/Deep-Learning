import copy

class NeuralNetwork:

    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, value):
        self._phase = value

    def forward(self):

        self.input_tensor , self.label_tensor = self.data_layer.next()

        y_hat = self.input_tensor
        #
        #print(self.layers)
        for layer in self.layers:

            y_hat = layer.forward(y_hat)

            #y_hats.append(y_hat)
        loss = self.loss_layer.forward(y_hat, self.label_tensor)

        #print(y_hats)
        return loss

    def backward(self):

        error_tensor = self.loss_layer.backward(self.label_tensor)

        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)





    def append_layer(self, layer):

        #print(layer)
        if layer.trainable is True:
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)

        self.layers.append(layer)

    def train(self, iterations):
        self._phase = True
        for layer in self.layers:
            layer.testing_phase = self._phase

        for i in range(iterations):
            self.loss.append(self.forward())
            self.backward()




    def test(self, input_tensor):

        self._phase = False
        for layer in self.layers:
            layer.testing_phase = self._phase

        y_hat =  input_tensor
        for layer in self.layers:
            y_hat = layer.forward(y_hat)

        return y_hat

