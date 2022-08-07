import numpy as np

from autograd import variable
import autograd
from autograd.node import Node


class add(Node):
    def __init__(self, x, y):
        super(add, self).__init__([x, y])

    def forward(self):
        x, y = self.get_incoming_nodes()
        self.output = np.add(x.data, y.data)
        return self.output

    def backward(self, with_respect):
        with_respect.gradients += np.ones(with_respect.shape)
        return variable.Variable(with_respect.gradients)


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
            with_respect.gradients += np.ones(x.shape)
        elif with_respect is y:
            with_respect.gradients += -np.ones(y.shape)
        return variable.Variable(with_respect.gradients)


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
            with_respect.gradients += variable_2.data
        elif with_respect is variable_2:
            with_respect.gradients += variable_1.data
        return variable.Variable(with_respect.gradients)


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
            with_respect.gradients += variable_2.data
        elif with_respect is variable_2:
            with_respect.gradients += variable_1.data
        return variable.Variable(with_respect.gradients)


class matmul(Node):
    def __init__(self, x, y):
        super().__init__([x, y])

    def forward(self):
        x, y = self.get_incoming_nodes()
        self.output = np.matmul(x.data, y.data)
        return self.output

    def backward(self, with_respect):
        variable_1, variable_2 = self.get_incoming_nodes()
        if with_respect is variable_1:
            with_respect.gradients += variable_2.data
        elif with_respect is variable_2:
            with_respect.gradients += variable_1.data
        return variable.Variable(with_respect.gradients)


class sum(Node):
    def __init__(self, x):
        super().__init__([x])

    def forward(self):
        self.output = self.get_incoming_nodes().data
        return self.output

    def backward(self, with_respect):
        x = self.get_incoming_nodes()
        if with_respect is x:
            with_respect.gradients += np.ones(x.shape)
        else:
            with_respect.gradients += np.zeros(x.shape)
        return variable.Variable(with_respect.gradients)


class power(Node):
    def __init__(self, x, p):
        super().__init__([x])
        self.p = p

    def forward(self):
        self.output = np.power(self.get_incoming_nodes().data, self.p)
        return self.output

    def backward(self, with_respect):
        if with_respect:
            with_respect.gradients += self.p * with_respect.data ** (self.p - 1)
        return variable.Variable(with_respect.gradients)


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
            with_respect.gradients += variable_2.data ** -1
        else:
            with_respect.gradients += variable_1.data * (-1 * variable_2.data ** -2)
        return variable.Variable(with_respect.gradients)


class exp(Node):
    def __init__(self, x):
        super().__init__([x])

    def forward(self):
        self.output = np.exp(self.get_incoming_nodes().data)
        return self.output

    def backward(self, with_respect):
        # self.data == self.forward() but cached
        with_respect.gradients += self.data
        return variable.Variable(with_respect.gradients)


class sigmoid(Node):
    def __init__(self, x):
        super().__init__([])
        self.exp_op = exp(-x)
        self.add_op = add(variable.Variable(1.0), self.exp_op)
        # Here the divide operation is stored in the self.output_node
        # because the divide operation is a nested operation (it's inside the sigmoid class/operation)
        # so we must specify which nested operation computes the final output for the sigmoid
        self.output_node = divide(variable.Variable(1.0), self.add_op)

    def forward(self):
        self.output = self.output_node.forward()
        return self.output

    def backward(self, with_respect):
        with_respect.gradients += self.output_node.backward(with_respect).data
        return variable.Variable(with_respect.gradients)

class relu(Node):
    def __init__(self, x):
        super().__init__([x])
    
    def forward(self):
        x = self.get_incoming_nodes()
        self.output = np.maximum(x.data, 0.)
        return self.output

    def backward(self, with_respect):
        out = np.sum(self.gradients, axis=1, keepdims=True)
        out = np.repeat(out, self.gradients.shape[0], 1)
        out = np.transpose(out, (1, 0))
        with_respect.gradients = out * (self.data > 0)
        return autograd.Variable(out)


class sin(Node):
    def __init__(self, x):
        super().__init__([x])

    def forward(self):
        self.output = np.sin(self.get_incoming_nodes().data)
        return self.output

    def backward(self, with_respect):
        with_respect.gradients += np.cos(self.get_incoming_nodes().data)
        return variable.Variable(with_respect.gradients)


class cos(Node):
    def __init__(self, x):
        super().__init__([x])

    def forward(self):
        self.output = np.cos(self.get_incoming_nodes().data)
        return self.output

    def backward(self, with_respect):
        with_respect.gradients += -np.sin(self.get_incoming_nodes().data)
        return variable.Variable(with_respect.gradients)


class sinh(Node):
    def __init__(self, x):
        super().__init__([x])

    def forward(self):
        self.output = np.sinh(self.get_incoming_nodes().data)
        return self.output

    def backward(self, with_respect):
        with_respect.gradients += np.cosh(self.get_incoming_nodes().data)
        return variable.Variable(with_respect.gradients)


class cosh(Node):
    def __init__(self, x):
        super().__init__([x])

    def forward(self):
        self.output = np.cosh(self.get_incoming_nodes().data)
        return self.output

    def backward(self, with_respect):
        with_respect.gradients += np.sinh(self.get_incoming_nodes().data)
        return variable.Variable(with_respect.gradients)
