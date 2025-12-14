import math
import random

# ------------------ Math utilities ------------------

def dot(a, b):
    # vector · vector OR vector · matrix
    if isinstance(b[0], list):
        return [sum(a[i] * b[i][j] for i in range(len(a))) for j in range(len(b[0]))]
    return sum(a[i] * b[i] for i in range(len(a)))

def add_vec(a, b):
    return [a[i] + b[i] for i in range(len(a))]

def outer(a, b):
    return [[a[i] * b[j] for j in range(len(b))] for i in range(len(a))]


# ------------------ Activation functions ------------------

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_vec(v):
    return [sigmoid(x) for x in v]

def sigmoid_derivative(y):
    return y * (1 - y)

# ------------------ Neural Network ------------------

class TinyNN:
    def __init__(self, n_input, n_hidden, n_output, lr=0.7):
        self.lr = lr

        self.W1 = [[random.uniform(-1, 1) for _ in range(n_hidden)] for _ in range(n_input)]
        self.b1 = [random.uniform(-1, 1) for _ in range(n_hidden)]

        self.W2 = [[random.uniform(-1, 1) for _ in range(n_output)] for _ in range(n_hidden)]
        self.b2 = [random.uniform(-1, 1) for _ in range(n_output)]

    def forward(self, x):
        z1 = add_vec(dot(x, self.W1), self.b1)
        a1 = sigmoid_vec(z1)

        z2 = add_vec(dot(a1, self.W2), self.b2)
        a2 = sigmoid_vec(z2)

        return x, a1, a2

    def train(self, X, Y, epochs=5000):
        for _ in range(epochs):
            for x, y in zip(X, Y):
                self._backprop(x, y)

    def _backprop(self, x, y):
        # forward
        x, a1, a2 = self.forward(x)

        # output error
        delta2 = [(y[i] - a2[i]) * sigmoid_derivative(a2[i]) for i in range(len(y))]

        # update W2, b2
        for i in range(len(self.W2)):
            for j in range(len(self.W2[0])):
                self.W2[i][j] += self.lr * a1[i] * delta2[j]
        for j in range(len(self.b2)):
            self.b2[j] += self.lr * delta2[j]

        # hidden error
        delta1 = []
        for i in range(len(a1)):
            error = sum(self.W2[i][j] * delta2[j] for j in range(len(delta2)))
            delta1.append(error * sigmoid_derivative(a1[i]))

        # update W1, b1
        for i in range(len(self.W1)):
            for j in range(len(self.W1[0])):
                self.W1[i][j] += self.lr * x[i] * delta1[j]
        for j in range(len(self.b1)):
            self.b1[j] += self.lr * delta1[j]

    def predict(self, x):
        _, _, output = self.forward(x)
        return output



if __name__ == "__main__":
    X = [[0,0], [0,1], [1,0], [1,1]]
    Y = [[0], [1], [1], [0]]

    nn = TinyNN(2, 4, 1)
    nn.train(X, Y)

    for x in X:
        print(x, nn.predict(x))
