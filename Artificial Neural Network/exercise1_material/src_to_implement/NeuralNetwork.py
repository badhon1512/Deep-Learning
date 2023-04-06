import copy

class NeuralNetwork:

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None

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
        if layer.trainable is True:
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)

    def train(self, iterations):

        self.input_tensor, self.label_tensor = self.data_layer.next()

        for i in range(iterations):
            self.loss.append(self.forward())
            self.backward()




    def test(self, input_tensor):

        y_hat =  input_tensor
        for layer in self.layers:
            y_hat = layer.forward(y_hat)

        return y_hat

