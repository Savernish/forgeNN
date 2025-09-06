class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        return Value(self.data + other.data, (self, other), _op='+')

    def __mul__(self, other):
        return Value(self.data * other.data, (self, other), _op='*')

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
    
    def __pow__(self, other): # self ** other
        return Value(self.data ** other, (self,), _op=f'**{other}')

