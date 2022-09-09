from autograd import primitive_ops, variable


def check_input_type(func):
    def wrapper(self, x):
        x = x if isinstance(x, OperationsMixin) else variable.Variable(x)
        return func(self, x)

    return wrapper


class OperationsMixin:
    @check_input_type
    def __add__(self, x):
        return primitive_ops.add(self, x)

    @check_input_type
    def __mul__(self, x):
        return primitive_ops.multiply(self, x)

    @check_input_type
    def __div__(self, x):
        return primitive_ops.divide(self, x)

    @check_input_type
    def __truediv__(self, x):
        return primitive_ops.divide(self, x)

    @check_input_type
    def __sub__(self, x):
        return primitive_ops.subtract(self, x)

    @check_input_type
    def __radd__(self, x):
        return primitive_ops.add(self, x)

    @check_input_type
    def __rmul__(self, x):
        return primitive_ops.multiply(self, x)

    @check_input_type
    def __rsub__(self, x):
        return primitive_ops.subtract(self, x)

    def __pow__(self, x):
        return primitive_ops.power(self, x)

    def __neg__(self):
        return primitive_ops.multiply(variable.Variable(-1), self)

    @check_input_type
    def add(self, x):
        return primitive_ops.add(self, x)

    @check_input_type
    def subtract(self, x):
        return primitive_ops.subtract(self, x)

    @check_input_type
    def multiply(self, x):
        return primitive_ops.multiply(self, x)

    @check_input_type
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

    def cosh(self):
        return primitive_ops.cosh(self)

    def sinh(self):
        return primitive_ops.sinh(self)

    def __hash__(self):
        return hash(id(self))
