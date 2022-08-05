from functools import total_ordering

from autograd import primitive_ops, variable


@total_ordering
class OperationsMixin:
    def __add__(self, x):
        return primitive_ops.add(self, x)

    def __mul__(self, x):
        return primitive_ops.multiply(self, x)

    def __div__(self, x):
        return primitive_ops.divide(self, x)

    def __pow__(self, x):
        return primitive_ops.power(self, x)

    def __sub__(self, x):
        return primitive_ops.subtract(self, x)

    def __lt__(self, x):
        return self.data < x.data

    def __eq__(self, x):
        return self.data == x.data

    def __le__(self, x):
        return self.data <= x.data

    def __ge__(self, x):
        return self.data >= x.data

    def __ne__(self, x):
        return self.data != x.data

    def __neg__(self):
        return primitive_ops.multiply(variable.Variable(-1), self)

    def add(self, x):
        return primitive_ops.add(self, x)

    def subtract(self, x):
        return primitive_ops.subtract(self, x)

    def multiply(self, x):
        return primitive_ops.multiply(self, x)

    def divide(self, x):
        return primitive_ops.divide(self, x)

    def power(self, x):
        return primitive_ops.power(self, x)

    def exp(self):
        return primitive_ops.exp(self)

    def sin(self):
        return primitive_ops.sin(self)

    def cos(self):
        return primitive_ops.cos(self)

    def compute_gradients(self):
        return self.outcoming_nodes[-1].compute_gradients(with_respect=self)
