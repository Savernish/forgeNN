from forgeNN.core import Value
from forgeNN.network import Neuron, Layer, MLP

if __name__ == "__main__":
    a = Value(2.0)
    b = Value(3.0)
    c = a * (b + a)
    print(c)
    print(c._prev) # Output: (Value(6.0), Value(2.0))
    print(c._op)   # Output: '+'
