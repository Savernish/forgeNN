from .functions.activation import RELU, LRELU, TANH, SIGMOID, SWISH

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.grad = 0.0
        self._backward = lambda: None

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), _op='+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), _op='*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        
        return out

    def __pow__(self, other): # self ** other
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), _op=f'**{other}')
        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward
        return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()


    def __repr__(self):
        #return f"Value(data={self.data})"
        return f"{self.data}"
    
    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1
    
    def apply_activation(self, activation_class, *args, **kwargs):
        """Apply an activation function from the functions library"""
        # Forward pass using the activation class
        activated_data = activation_class.forward(self.data, *args, **kwargs)
        out = Value(activated_data, (self,), f'{activation_class.__name__}')
        
        def _backward():
            # Backward pass using the activation class
            grad_input = activation_class.backward(self.data, *args, **kwargs)
            self.grad += grad_input * out.grad
        out._backward = _backward
        
        return out
    
    def relu(self):
        """Apply ReLU activation using the RELU class from functions library"""
        return self.apply_activation(RELU)
    
    def lrelu(self, alpha=0.01):
        """Apply Leaky ReLU activation using the LRELU class from functions library"""
        return self.apply_activation(LRELU, alpha=alpha)
    
    def tanh(self):
        """Apply Tanh activation using the TANH class from functions library"""
        return self.apply_activation(TANH)
    
    def sigmoid(self):
        """Apply Sigmoid activation using the SIGMOID class from functions library"""
        return self.apply_activation(SIGMOID)
    
    def swish(self):
        """Apply Swish activation using the SWISH class from functions library"""
        return self.apply_activation(SWISH)

