# Neural Network From Scratch (Single File)

import math
import random

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(y):
    return y * (1 - y)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, lr=0.7):
        self.lr = lr

        self.W1 = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(input_size)]
        self.b1 = [random.uniform(-1, 1) for _ in range(hidden_size)]

        self.W2 = [[random.uniform(-1, 1) for _ in range(output_size)] for _ in range(hidden_size)]
        self.b2 = [random.uniform(-1, 1) for _ in range(output_size)]

    def forward(self, x):
        z1 = [sum(x[i] * self.W1[i][j] for i in range(len(x))) + self.b1[j]
              for j in range(len(self.b1))]
        a1 = [sigmoid(z) for z in z1]

        z2 = [sum(a1[i] * self.W2[i][j] for i in range(len(a1))) + self.b2[j]
              for j in range(len(self.b2))]
        a2 = [sigmoid(z) for z in z2]

        return x, a1, a2

    def train(self, X, Y, epochs=5000):
        for epoch in range(epochs):
            for x, y in zip(X, Y):
                self.backprop(x, y)
            if epoch % 1000 == 0:
                print("Epoch", epoch)

    def backprop(self, x, y):
        x, a1, a2 = self.forward(x)

        delta2 = [(y[i] - a2[i]) * sigmoid_derivative(a2[i]) for i in range(len(y))]

        for i in range(len(self.W2)):
            for j in range(len(self.W2[0])):
                self.W2[i][j] += self.lr * a1[i] * delta2[j]

        for j in range(len(self.b2)):
            self.b2[j] += self.lr * delta2[j]

        delta1 = []
        for i in range(len(a1)):
            error = sum(self.W2[i][j] * delta2[j] for j in range(len(delta2)))
            delta1.append(error * sigmoid_derivative(a1[i]))

        for i in range(len(self.W1)):
            for j in range(len(self.W1[0])):
                self.W1[i][j] += self.lr * x[i] * delta1[j]

        for j in range(len(self.b1)):
            self.b1[j] += self.lr * delta1[j]

    def predict(self, x):
        _, _, output = self.forward(x)
        return output


if __name__ == "__main__":
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    Y = [[0], [1], [1], [0]]

    nn = NeuralNetwork(2, 4, 1)
    nn.train(X, Y)

    print("\nFinal Predictions:")
    for x in X:
        print(x, nn.predict(x))
