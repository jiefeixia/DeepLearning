"""
Follow the instructions provided in the writeup to completely
implement the class specifications for a basic MLP, optimizer, .
You will be able to test each section individually by submitting
to autolab after implementing what is required for that section
-- do not worry if some methods required are not implemented yet.

Notes:

The __call__ method is a special reserved method in
python that defines the behaviour of an object when it is
used as a function. For example, take the Linear activation
function whose implementation has been provided.

# >>> activation = Identity()
# >>> activation(3)
# 3
# >>> activation.forward(3)
# 3
"""

# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)
import numpy as np
import os


class Activation(object):
    """
    Interface for activation functions (non-linearities).

    In all implementations, the state attribute must contain the result, i.e. the output of forward (it will be tested).

    shape of x: (batch_size, layer_width)
    """

    # No additional work is needed for this class, as it acts like an abstract base class for the others

    def __init__(self):
        self.state = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class Identity(Activation):
    """
    Identity function (already implemented).
    """

    # This class is a gimme as it is already implemented for you as an example

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        self.state = x
        return x

    def derivative(self):
        return 1.0


class Sigmoid(Activation):
    """
    Sigmoid non-linearity
    """

    # Remember do not change the function signatures as those are needed to stay the same for AL

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        self.state = 1 / (1 + np.exp(-x))
        return self.state

    def derivative(self):
        return self.state * (1 - self.state)


class Tanh(Activation):
    """
    Tanh non-linearity
    """

    # This one's all you!

    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        self.state = np.tanh(x)
        return self.state

    def derivative(self):
        return 1 - self.state ** 2


class ReLU(Activation):
    """
    ReLU non-linearity
    """

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        self.state = x
        self.state[self.state < 0] = 0
        return self.state

    def derivative(self):
        d = np.ones(self.state.shape)
        d[self.state == 0] = 0
        return d


# Ok now things get decidedly more interesting. The following Criterion class
# will be used again as the basis for a number of loss functions (which are in the
# form of classes so that they can be exchanged easily (it's how PyTorch and other
# ML libraries do it))


class Criterion(object):
    """
    Interface for loss functions.
    """

    # Nothing needs done to this class, it's used by the following Criterion classes

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class SoftmaxCrossEntropy(Criterion):
    """
    Softmax loss
    """

    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()
        self.sm = None

    def forward(self, x, y):
        self.logits = x
        self.labels = y
        x_max = np.max(x, axis=1)

        self.sm = (np.exp(x.T - x_max) / np.sum(np.exp(x.T - x_max), axis=0)).T
        return -np.sum(y * np.log(self.sm), axis=1)

    def derivative(self):
        return self.sm - self.labels


class BatchNorm(object):

    def __init__(self, fan_in, alpha=0.9):
        # You shouldn't need to edit anything in init

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested
        self.var = np.ones((1, fan_in))
        self.mean = np.zeros((1, fan_in))

        self.gamma = np.ones((1, fan_in))
        self.dgamma = np.zeros((1, fan_in))

        self.beta = np.zeros((1, fan_in))
        self.dbeta = np.zeros((1, fan_in))

        # inference parameters
        self.running_mean = np.zeros((1, fan_in))
        self.running_var = np.ones((1, fan_in))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):
        """
        :param x: shape(batch_size, units)
        :param eval:
        :return: y shape(batch_size, units)
        """
        self.x = x

        if eval:
            self.norm = (x - self.running_mean) / (self.running_var + self.eps)**(1/2)
            self.out = self.gamma * self.norm + self.beta
            return self.out

        self.mean = np.mean(x, axis=0)
        self.var = np.var(x, axis=0)
        self.norm = (x - self.mean) / (self.var + self.eps)**(1/2)
        self.out = self.gamma * self.norm + self.beta

        # update running batch statistics
        self.running_mean = self.alpha * self.running_mean + (1 - self.alpha) * self.mean
        self.running_var = self.alpha * self.running_var + (1 - self.alpha) * self.var

        return self.out

    def backward(self, delta):
        """
        :param delta: dL/dz_hat, shape(batch_size, units)
        :return: dL/dz, shape(batch_size, units)
        """

        batch_size = delta.shape[0]

        self.dgamma = np.sum(delta * self.out, axis=0)
        self.dbeta = np.sum(delta, axis=0)

        dl_dnorm = delta * self.gamma  # (batch_size, unit)
        dl_dvar = - 1 / 2 * np.sum(dl_dnorm
                                   * (self.x - self.mean)
                                   * (self.var + self.eps)**(-3/2),
                                   axis=0)
        dl_dmu = - np.sum(dl_dnorm * (self.var + self.eps)**(-1/2), axis=0) \
                 - 2 / batch_size * dl_dvar * np.sum(self.x - self.mean, axis=0)

        return (dl_dnorm * (self.var + self.eps) ** (-1/2)
                + dl_dvar * 2 / batch_size * (self.x - self.mean)
                + dl_dmu / batch_size)


# These are both easy one-liners, don't over-think them
def random_normal_weight_init(d0, d1):
    return np.random.randn(d0, d1)


def zeros_bias_init(d):
    return np.zeros(d)


class MLP(object):
    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn, bias_init_fn, criterion, lr,
                 momentum=0.0, num_bn_layers=0):
        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly
        input_hidden_output = [input_size] + hiddens + [output_size]
        self.W = [weight_init_fn(input_hidden_output[i], input_hidden_output[i + 1])
                  for i in range(self.nlayers)]  # shape of W: [ndarray(output, input), layer]
        self.b = [bias_init_fn(input_hidden_output[i + 1]) for i in range(self.nlayers)]
        self.delta_b = [np.zeros(self.b[layer].shape) for layer in range(self.nlayers)]
        self.delta_W = [np.zeros(self.W[layer].shape) for layer in range(self.nlayers)]

        # if batch norm, add batch norm parameters
        if self.bn:
            self.bn_layers = [BatchNorm(input_hidden_output[i]) for i in range(1, num_bn_layers + 1)]

        # Feel free to add any other attributes useful to your implementation (input, output, ...)
        self.input = None
        self.db = [np.zeros(self.b[layer].shape) for layer in range(self.nlayers)]
        self.dW = [np.zeros(self.W[layer].shape) for layer in range(self.nlayers)]

    def forward(self, x):
        """
        :param x:
        :return: output: f(WX+b)
        """
        self.input = x

        # layers with BN
        input_ = self.input
        for layer in range(self.num_bn_layers):
                z = input_.dot(self.W[layer]) + self.b[layer]
                input_ = self.activations[layer](self.bn_layers[layer](z) if self.train_mode else
                                                 self.bn_layers[layer](z, eval=True))

        # layers without BN
        for layer in range(self.num_bn_layers, self.nlayers):
            z = input_.dot(self.W[layer]) + self.b[layer]
            input_ = self.activations[layer](z)

        return self.activations[-1].state

    def zero_grads(self):
        self.db = [np.zeros(self.b[layer].shape) for layer in range(self.nlayers)]
        self.dW = [np.zeros(self.W[layer].shape) for layer in range(self.nlayers)]
        if self.bn:
            for bn in self.bn_layers:
                bn.dbeta = np.zeros(bn.dbeta.shape)
                bn.dgamma = np.zeros(bn.dgamma.shape)

    def step(self):
        for layer in range(self.nlayers):
            self.delta_W[layer] = self.momentum * self.delta_W[layer] - self.lr * self.dW[layer]
            self.delta_b[layer] = self.momentum * self.delta_b[layer] - self.lr * self.db[layer]

            self.W[layer] += self.delta_W[layer]
            self.b[layer] += self.delta_b[layer]
        if self.bn:
            for bn in self.bn_layers:
                bn.gamma -= self.lr * bn.dgamma
                bn.beta -= self.lr * bn.dbeta

    def backward(self, labels):
        batch_size = labels.shape[0]
        # Output layer
        _loss = self.criterion(self.activations[-1].state, labels)
        dl_dy = self.criterion.derivative()

        # without BM
        for layer in range(self.nlayers - 1, self.num_bn_layers - 1, -1):
            dl_dz = self.activations[layer].derivative() * dl_dy
            self.db[layer] = np.average(dl_dz, axis=0)
            if layer > 0:
                self.dW[layer] = np.average([np.outer(self.activations[layer - 1].state[batch],
                                                      dl_dz[batch])
                                             for batch in range(batch_size)],
                                            axis=0)
                dl_dy = dl_dz.dot(self.W[layer].T)

            else:  # last layer
                self.dW[layer] = np.average([np.outer(self.input[batch],
                                                      dl_dz[batch])
                                             for batch in range(batch_size)],
                                            axis=0)

        # with BM
        for layer in range(self.num_bn_layers - 1, -1, -1):
            dl_dz = self.bn_layers[layer].backward(self.activations[layer].derivative() * dl_dy)
            self.db[layer] = np.average(dl_dz, axis=0)
            if layer > 0:
                self.dW[layer] = np.average([np.outer(self.activations[layer - 1].state[batch],
                                                      dl_dz[batch])
                                             for batch in range(batch_size)],
                                            axis=0)
                dl_dy = dl_dz.dot(self.W[layer].T)

            else:  # last layer
                self.dW[layer] = np.average([np.outer(self.input[batch],
                                                      dl_dz[batch])
                                             for batch in range(batch_size)],
                                            axis=0)

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False


def get_training_stats(mlp, dset, nepochs, batch_size):
    train, val, test = dset
    trainx, trainy = train
    valx, valy = val
    testx, testy = test

    idxs = np.arange(len(trainx))

    training_losses = []
    training_errors = []
    validation_losses = []
    validation_errors = []

    # set up

    for e in range(nepochs):

        np.random.shuffle(idxs)
        training_errors_epoch = []
        training_losses_epoch = []
        validation_errors_epoch = []
        validation_losses_epoch = []

        for b in range(0, len(trainx), batch_size):
            if b == 0:
                b_pre = b
                continue
            x = trainx[idxs[b_pre:b]]
            y = trainy[idxs[b_pre:b]]
            b_pre = b

            mlp.train()
            mlp.zero_grads()
            output = mlp.forward(x)
            mlp.backward(y)
            mlp.step()

            training_losses_epoch += [np.average(mlp.criterion(output, y))]
            pred = np.argmax(output, axis=1)
            label = np.argmax(y, axis=1)
            training_errors_epoch += [1 - np.sum(pred == label) / y.shape[0]]

        for b in range(0, len(valx), batch_size):
            if b == 0:
                b_pre = b
                continue
            idxs_v = np.arange(len(valx))
            x = valx[idxs_v[b_pre:b]]
            y = valy[idxs_v[b_pre:b]]
            b_pre = b

            mlp.eval()
            output = mlp.forward(x)

            validation_losses_epoch += [np.average(mlp.criterion(output, y))]
            pred = np.argmax(output, axis=1)
            label = np.argmax(y, axis=1)
            validation_errors_epoch += [1 - np.sum(pred == label) / y.shape[0]]

        # Accumulate data
        training_errors += [np.average(training_errors_epoch)]
        training_losses += [np.average(training_losses_epoch)]
        validation_errors += [np.average(validation_errors_epoch)]
        validation_losses += [np.average(validation_losses_epoch)]

        if e % 10 == 0:
            print("\n\n========= epoch" + str(e) + " ==========")
            print("tra error: " + str(training_errors[e]))
            print("val error: " + str(validation_errors[e]))
            print("tra loss: " + str(training_losses[e]))
            print("val loss: " + str(validation_losses[e]))
    # Cleanup ...

    test_losses_batch = []
    test_errors_batch = []
    for b in range(0, len(testx), batch_size):
        if b == 0:
            b_pre = b
            pass
        x = testx[b_pre:b]
        y = testy[b_pre:b]

        mlp.eval()
        output = mlp.forward(x)

        test_losses_batch += [np.average(mlp.criterion(output, y))]
        pred = np.argmax(output, axis=1)
        label = np.argmax(y, axis=1)
        test_errors_batch += [1 - np.sum(pred == label) / y.shape[0]]

    # Return results
    test_error = np.average(test_losses_batch)
    test_loss = np.average(test_errors_batch)

    print("test loss: " + str(test_loss))
    print("test error: " + str(test_error))

    return (training_losses, training_errors, validation_losses, validation_errors)
