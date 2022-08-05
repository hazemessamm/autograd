import numpy as np

from autograd import variable
from autograd.node import Node


class add(Node):
    def __init__(self, x, y):
        super(add, self).__init__([x, y])

    def forward(self):
        x, y = self.get_incoming_nodes()
        self.output = np.add(x.data, y.data)
        return self.output

    def backward(self, with_respect):
        return variable.Variable(np.ones(with_respect.shape))


class subtract(Node):
    def __init__(self, x, y):
        super(subtract, self).__init__([x, y])

    def forward(self):
        x, y = self.get_incoming_nodes()
        self.output = np.subtract(x.data, y.data)
        return self.output

    def backward(self, with_respect):
        x, y = self.get_incoming_nodes()
        if with_respect is x:
            derivative = np.ones(x.shape)
        elif with_respect is y:
            derivative = -np.ones(y.shape)
        return variable.Variable(derivative)


class multiply(Node):
    def __init__(self, x, y):
        super(multiply, self).__init__([x, y])

    def forward(self):
        x, y = self.get_incoming_nodes()
        self.output = np.multiply(x.data, y.data)
        return self.output

    def backward(self, with_respect):
        variable_1, variable_2 = self.get_incoming_nodes()
        if with_respect is variable_1:
            derivative = variable_2.data
        elif with_respect is variable_2:
            derivative = variable_1.data
        else:
            derivative = 0.0
        return variable.Variable(derivative)


class dot(Node):
    def __init__(self, x, y):
        super().__init__([x, y])

    def forward(self):
        x, y = self.get_incoming_nodes()
        self.output = np.dot(x.data, y.data)
        return self.output

    def backward(self, with_respect):
        variable_1, variable_2 = self.get_incoming_nodes()
        if with_respect is variable_1:
            derivative = variable_2.data
        elif with_respect is variable_2:
            derivative = variable_1.data
        else:
            derivative = 0.0
        return variable.Variable(derivative)


class sum(Node):
    def __init__(self, x):
        super().__init__([x])

    def forward(self):
        self.output = self.get_incoming_nodes().data
        return self.output

    def backward(self, with_respect):
        x = self.get_incoming_nodes()
        if with_respect is x:
            derivative = np.ones(x.shape)
        else:
            derivative = np.zeros(x.shape)
        return variable.Variable(derivative)


class power(Node):
    def __init__(self, x, p):
        super().__init__([x])
        self.p = p

    def forward(self):
        self.output = np.power(self.get_incoming_nodes().data, self.p)
        return self.output

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
        x, y = self.get_incoming_nodes()
        self.output = np.divide(x.data, y.data)
        return self.output

    def backward(self, with_respect):
        variable_1, variable_2 = self.get_incoming_nodes()
        if with_respect is variable_1:
            derivative = variable_2.data ** -1
        else:
            derivative = variable_1.data * (-1 * variable_2.data ** -2)
        return variable.Variable(derivative)


class exp(Node):
    def __init__(self, x):
        super().__init__([x])

    def forward(self):
        self.output = np.exp(self.get_incoming_nodes().data)
        return self.output

    def backward(self, with_respect):
        derivative = self.forward()
        return variable.Variable(derivative)


class sigmoid(Node):
    def __init__(self, x):
        super().__init__([])
        self.exp_op = exp(-x)
        self.add_op = add(variable.Variable(1.0), self.exp_op)
        self.output_node = divide(variable.Variable(1.0), self.add_op)

    def forward(self):
        self.output = self.output_node.forward()
        return self.output

    def backward(self, with_respect):
        return variable.Variable(self.output_node.backward(with_respect).data)


class sin(Node):
    def __init__(self, x):
        super().__init__([x])

    def forward(self):
        self.output = np.sin(self.get_incoming_nodes().data)
        return self.output

    def backward(self):
        derivative = np.cos(self.get_incoming_nodes().data)
        return variable.Variable(derivative)


class cos(Node):
    def __init__(self, x):
        super().__init__([x])

    def forward(self):
        self.output = np.cos(self.get_incoming_nodes().data)
        return self.output

    def backward(self):
        derivative = -np.sin(self.get_incoming_nodes().data)
        return variable.Variable(derivative)


class sinh(Node):
    def __init__(self, x):
        super().__init__([x])

    def forward(self):
        self.output = np.sinh(self.get_incoming_nodes().data)
        return self.output

    def backward(self):
        derivative = np.cosh(self.get_incoming_nodes().data)
        return variable.Variable(derivative)


class cosh(Node):
    def __init__(self, x):
        super().__init__([x])

    def forward(self):
        self.output = np.cosh(self.get_incoming_nodes().data)
        return self.output

    def backward(self):
        derivative = np.sinh(self.get_incoming_nodes().data)
        return variable.Variable(derivative)
