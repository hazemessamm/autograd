from autograd import primitive_ops, variable

class OperationsMixin:
    def __add__(self, x):
        x = x if isinstance(x, OperationsMixin) else variable.Variable(x)
        return primitive_ops.add(self, x)

    def __mul__(self, x):
        x = x if isinstance(x, OperationsMixin) else variable.Variable(x)
        return primitive_ops.multiply(self, x)

    def __div__(self, x):
        x = x if isinstance(x, OperationsMixin) else variable.Variable(x)
        return primitive_ops.divide(self, x)

    def __sub__(self, x):
        x = x if isinstance(x, OperationsMixin) else variable.Variable(x)
        return primitive_ops.subtract(self, x)

    def __pow__(self, x):
        return primitive_ops.power(self, x)

    def __neg__(self):
        return primitive_ops.multiply(variable.Variable(-1), self)

    def add(self, x):
        x = x if isinstance(x, OperationsMixin) else variable.Variable(x)
        return primitive_ops.add(self, x)

    def subtract(self, x):
        x = x if isinstance(x, OperationsMixin) else variable.Variable(x)
        return primitive_ops.subtract(self, x)

    def multiply(self, x):
        x = x if isinstance(x, OperationsMixin) else variable.Variable(x)
        return primitive_ops.multiply(self, x)

    def divide(self, x):
        x = x if isinstance(x, OperationsMixin) else variable.Variable(x)
        return primitive_ops.divide(self, x)

    def power(self, x):
        return primitive_ops.power(self, x)

    def exp(self):
        return primitive_ops.exp(self)

    def sin(self):
        return primitive_ops.sin(self)

    def cos(self):
        return primitive_ops.cos(self)

    def cosh(self):
        return primitive_ops.cosh(self)

    def sinh(self):
        return primitive_ops.sinh(self)

    def compute_gradients(self, save_gradients=True):
        gradient = self.outcoming_nodes[-1].backward(with_respect=self)
        if save_gradients:
            self.gradients = gradient
        return gradient
    
    def __hash__(self):
        return hash(id(self))