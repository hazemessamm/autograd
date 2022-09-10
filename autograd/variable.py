import threading
import weakref

import numpy as np

from autograd.exceptions import PlaceholderNotAssignedError
from autograd.ops_mixin import OperationsMixin


class Leaf:
    instances = weakref.WeakSet()
    num_instances = 0
    def __init__(self, data, name: str = None):
        if data is not None:
            data = np.array(data)
        self._data = data
        self.outcoming_nodes = []
        self.gradients = 0.
        Leaf.instances.add(self)

        with threading.Lock():
            Leaf.num_instances += 1
            self.counter = Leaf.num_instances
        
        if name is None:
            self.name = f'<{self.__class__.__name__.capitalize()}{self.counter}>'
        else:
            self.name = name

    @property
    def data(self):
        out = self._data
        if isinstance(out, Leaf):
            return out._data
        return out

    def is_placeholder(self):
        return getattr(self, '_assigned', False) == True

    @property
    def shape(self):
        if self.is_placeholder() and not self._assigned:
            raise ValueError("Placeholder is not initialized yet.")
        return self._data.shape

    def __repr__(self):
        return self.name


class Variable(Leaf, OperationsMixin):
    def __init__(self, data, **kwargs):
        if data is None:
            raise ValueError("Cannot assign `None` to data.")
        super(Variable, self).__init__(data, **kwargs)


class Placeholder(Leaf, OperationsMixin):
    def __init__(self, **kwargs):
        super().__init__(data=None, **kwargs)
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
        return self
