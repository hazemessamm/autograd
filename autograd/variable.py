import weakref

import numpy as np

from autograd.exceptions import PlaceholderNotAssignedError
from autograd.ops_mixin import OperationsMixin


class Leaf:
    instances = weakref.WeakSet()
    def __init__(self, data):
        if data is not None:
            data = np.array(data)
        self._data = data
        self.outcoming_nodes = []
        self.gradients = 0.
        Leaf.instances.add(self)

    @property
    def data(self):
        return self._data

    def is_placeholder(self):
        return getattr(self, '_assigned', False) == True

    @property
    def shape(self):
        if self.is_placeholder() and not self._assigned:
            raise ValueError("Placeholder is not initialized yet.")
        return self._data.shape

    def __repr__(self):
        return f"<{self.__class__.__name__.capitalize()}>"


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
            raise PlaceholderNotAssignedError(f"{self} is not Assigned yet to a value.")
        return self._data

    def assign(self, data):
        if data is None:
            raise ValueError("Cannot assign `None` to data.")
        self._data = np.array(data)
        self._assigned = True
