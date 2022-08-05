import numpy as np
from autograd.node import Node
from autograd import variable

class add(Node):
    def __init__(self, x, y):
        super(add, self).__init__([x, y])
    
    def forward(self):
        return np.add(self.incoming_nodes[0].data, self.incoming_nodes[1].data)

    def backward(self, with_respect):
        return variable.Variable(1.)

class subtract(Node):
    def __init__(self, x, y):
        super(subtract, self).__init__([x, y])
    
    def forward(self):
        return np.add(self.incoming_nodes[0].data, self.incoming_nodes[1].data)

    def backward(self, with_respect):
        return variable.Variable(1. if with_respect == self.incoming_nodes[0] else -1.)

class multiply(Node):
    def __init__(self, x, y):
        super(multiply, self).__init__([x, y])
    
    def forward(self):
        return np.multiply(self.incoming_nodes[0].data, self.incoming_nodes[1].data)
    
    def backward(self, with_respect):
        variable_1, variable_2 = self.incoming_nodes
        if with_respect == variable_1:
            return variable.Variable(variable_2.data)
        elif with_respect == variable_2:
            return variable.Variable(variable_1.data)
        else:
            return variable.Variable(0.)

class power(Node):
    def __init__(self, x, p):
        super().__init__([x])
        self.p = p
    
    def forward(self):
        return np.power(self.incoming_nodes[0].data, self.p)
    
    def backward(self, with_respect):
        if with_respect:
            return variable.Variable(self.p * with_respect.data ** (self.p-1))
        else:
            return variable.Variable(0.)

class divide(Node):
    def __init__(self, x, y):
        super().__init__([x, y])
    
    def forward(self):
        return np.divide(self.incoming_nodes[0].data, self.incoming_nodes[1].data)
    
    def backward(self, with_respect):
        if with_respect == self.incoming_nodes[0]:
            return variable.Variable(self.incoming_nodes[1].data ** -1)
        else:
            return variable.Variable(self.incoming_nodes[0].data * (-1 * self.incoming_nodes[1].data ** -2))

class exp(Node):
    def __init__(self, x):
        super().__init__([x])

    def forward(self):
        out = np.exp(self.incoming_nodes[0].data)
        return out

    def backward(self, with_respect):
        return variable.Variable(self.forward())


class sigmoid(Node):
    def __init__(self, x):
        super().__init__([x])
        self.exp = exp(-x)
        self.add = add(variable.Variable(1.), self.exp)
        self.div = divide(variable.Variable(1.), self.add)

    def forward(self):
        return self.div.forward()

    def backward(self, with_respect):
        return self.div.backward(with_respect)

class sin(Node):
    def __init__(self, x):
        super().__init__([x])
    
    def forward(self):
        return np.sin(self.incoming_nodes[0].data)
    
    def backward(self):
        return variable.Variable(np.cos(self.incoming_nodes[0].data))


class cos(Node):
    def __init__(self, x):
        super().__init__([x])
    
    def forward(self):
        return np.cos(self.incoming_nodes[0].data)
    
    def backward(self):
        return variable.Variable(-np.sin(self.incoming_nodes[0].data))
