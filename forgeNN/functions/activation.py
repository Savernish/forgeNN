# Activation functions for neural networks
import math

class RELU:
    @staticmethod
    def forward(x):
        return x if x > 0 else 0

    @staticmethod
    def backward(x):
        return 1.0 if x > 0 else 0.0
    

class LRELU:
    @staticmethod
    def forward(x, alpha=0.01):
        return x if x > 0 else alpha * x

    @staticmethod
    def backward(x, alpha=0.01):
        return 1.0 if x > 0 else alpha


class TANH:
    @staticmethod
    def forward(x):
        return math.tanh(x)

    @staticmethod
    def backward(x):
        t = math.tanh(x)
        return 1 - t * t


class SIGMOID:
    @staticmethod
    def forward(x):
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def backward(x):
        s = 1 / (1 + math.exp(-x))
        return s * (1 - s)


class SWISH:
    @staticmethod
    def forward(x):
        return x / (1 + math.exp(-x))

    @staticmethod
    def backward(x):
        s = 1 / (1 + math.exp(-x))  # sigmoid
        return s + x * s * (1 - s)