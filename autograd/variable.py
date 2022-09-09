import numpy as np

from autograd.exceptions import PlaceholderNotAssignedError
from autograd.ops_mixin import OperationsMixin


class Leaf:
    def __init__(self, data):
        if data is not None:
            data = np.array(data)
        self._data = data
        self.outcoming_nodes = []
        self.gradients = 0.

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        if getattr(self, "_assigned", False) and not self._assigned:
            raise ValueError("Placeholder is not initialized yet.")
        return self._data.shape

    def __repr__(self):
        return f"<{self.__class__.__name__.capitalize()} {self.data}>"


class Variable(Leaf, OperationsMixin):
    def __init__(self, data):
        if data is None:
            raise ValueError("Cannot assign `None` to data.")
        super(Variable, self).__init__(data)


class Placeholder(Leaf, OperationsMixin):
    def __init__(self):
        super().__init__(data=None)
        self._assigned = False

    @property
    def assigned(self):
        return self._assigned

    @property
    def data(self):
        if not self.assigned:
            error_msg = f"{self} is not Assigned yet to a value."
            raise PlaceholderNotAssignedError(error_msg)
        return self._data

    def assign(self, data):
        if data is None:
            raise ValueError("Cannot assign `None` to data.")
        self._data = np.array(data)
        self._assigned = True
