import numpy as np

from autograd import variable
from autograd.node import Node


class add(Node):
    def __init__(self, x, y):
        super(add, self).__init__([x, y])

    def forward(self):
        return np.add(self.incoming_nodes[0].data, self.incoming_nodes[1].data)

    def backward(self, with_respect):
        return variable.Variable(1.0)


class subtract(Node):
    def __init__(self, x, y):
        super(subtract, self).__init__([x, y])

    def forward(self):
        return np.add(self.incoming_nodes[0].data, self.incoming_nodes[1].data)

    def backward(self, with_respect):
        derivative = 1.0 if with_respect == self.incoming_nodes[0] else -1.0
        return variable.Variable(derivative)


class multiply(Node):
    def __init__(self, x, y):
        super(multiply, self).__init__([x, y])

    def forward(self):
        x, y = self.incoming_nodes[0].data, self.incoming_nodes[1].data
        return np.multiply(x, y)

    def backward(self, with_respect):
        variable_1, variable_2 = self.incoming_nodes
        if with_respect == variable_1:
            derivative = variable_2.data
        elif with_respect == variable_2:
            derivative = variable_1.data
        else:
            derivative = 0.0
        return variable.Variable(derivative)


class power(Node):
    def __init__(self, x, p):
        super().__init__([x])
        self.p = p

    def forward(self):
        return np.power(self.incoming_nodes[0].data, self.p)

    def backward(self, with_respect):
        if with_respect:
            derivative = self.p * with_respect.data ** (self.p - 1)
        else:
            derivative = 0.0
        return variable.Variable(derivative)


class divide(Node):
    def __init__(self, x, y):
        super().__init__([x, y])

    def forward(self):
        x, y = self.incoming_nodes[0].data, self.incoming_nodes[1].data
        return np.divide(x, y)

    def backward(self, with_respect):
        variable_1, variable_2 = self.incoming_nodes
        if with_respect == variable_1:
            derivative = variable_2.data ** -1
        else:
            derivative = variable_1.data * (-1 * variable_2.data ** -2)
        return variable.Variable(derivative)


class exp(Node):
    def __init__(self, x):
        super().__init__([x])

    def forward(self):
        out = np.exp(self.incoming_nodes[0].data)
        return out

    def backward(self, with_respect):
        derivative = self.forward()
        return variable.Variable(derivative)


class sigmoid(Node):
    def __init__(self, x):
        super().__init__([x])
        self.exp = exp(-x)
        self.add = add(variable.Variable(1.0), self.exp)
        self.div = divide(variable.Variable(1.0), self.add)

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
        derivative = np.cos(self.incoming_nodes[0].data)
        return variable.Variable(derivative)


class cos(Node):
    def __init__(self, x):
        super().__init__([x])

    def forward(self):
        return np.cos(self.incoming_nodes[0].data)

    def backward(self):
        derivative = -np.sin(self.incoming_nodes[0].data)
        return variable.Variable(derivative)


class sinh(Node):
    def __init__(self, x):
        super().__init__([x])

    def forward(self):
        return np.sinh(self.incoming_nodes[0].data)

    def backward(self):
        derivative = np.cosh(self.incoming_nodes[0].data)
        return variable.Variable(derivative)


class cosh(Node):
    def __init__(self, x):
        super().__init__([x])

    def forward(self):
        return np.cosh(self.incoming_nodes[0].data)

    def backward(self):
        derivative = np.sinh(self.incoming_nodes[0].data)
        return variable.Variable(derivative)
