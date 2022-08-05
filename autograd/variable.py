from autograd.ops_mixin import OperationsMixin
from autograd.exceptions import PlaceholderNotAssignedError
import numpy as np

class Leaf:
    def __init__(self, data):
        if data is not None:
            data = np.array(data)
        self._data = data
        self.outcoming_nodes = []
        self.gradients = None

    @property
    def data(self):
        return self._data

    def __repr__(self):
        return f"<{self.__class__.__name__.capitalize()} {self.data}>"
    
class Variable(Leaf, OperationsMixin):
    def __init__(self, data):
        if data is None:
            raise ValueError('Cannot assign `None` to data.')
        super(Variable, self).__init__(data)

class Placeholder(Leaf, OperationsMixin):
    def __init__(self):
        super().__init__(data=None)
        self._assigned = False

    @property
    def assigned(self): return self._assigned

    @property
    def data(self):
        if not self.assigned:
            raise PlaceholderNotAssignedError(f'{self} is not Assigned yet to a value.')
        return self._data

    def assign(self, data):
        if data is None:
            raise ValueError('Cannot assign `None` to data.')
        self._data = np.array(data)
        self._assigned = True

