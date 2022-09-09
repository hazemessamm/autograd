from typing import List, TypeVar, Union
from autograd.ops_mixin import OperationsMixin
from autograd.variable import Variable
import weakref
import abc

Node = TypeVar("Node", bound="Node")

class Node(abc.ABC, OperationsMixin):
    instances = weakref.WeakSet()
    def __init__(self, incoming_nodes: List[Node] = []):
        # incoming operations or variables
        self.incoming_nodes = incoming_nodes
        # outcoming operations or variables
        self.outcoming_nodes = []
        # for caching outputs
        self.output = None
        # for connecting nodes
        self._attach_to_outcoming_nodes()
        # caching gradients
        self.gradients = 0.
        self._nodes = None
        self.nested = False
        self.counter = len(Node.instances)
        Node.instances.add(self)

    def __setattr__(self, __name: str, __value: Node) -> None:
        if isinstance(__value, Node):
            if self._nodes is None:
                self._nodes = []
                self.nested = True
            self._nodes.append(__value)
        return super().__setattr__(__name, __value)

    @property
    def data(self):
        if self.output is not None:
            return self.output
        return self.forward()

    @property
    def shape(self):
        return self.data.shape

    def cache_output(self, output):
        self.output = output
        return output

    def get_incoming_nodes(self) -> Union[List[Node], Node]:
        '''Returns incoming nodes and also checks
        if an incoming node has nested nodes
        so it can return the last nested node'''

        if len(self.incoming_nodes) > 1:
            return self.incoming_nodes

        incoming_node = self.incoming_nodes[0]
        if not len(self.incoming_nodes) == 1:
            return self.incoming_nodes
        
        if isinstance(incoming_node, Node) and incoming_node.nested:
            return incoming_node._nodes[-1]
        else:
            return incoming_node

    def _attach_to_outcoming_nodes(self):
        for node in self.incoming_nodes:
            if self not in node.outcoming_nodes:
                node.outcoming_nodes.append(self)

        for n in self.incoming_nodes:
            if isinstance(n, Node) and n.nested:
                n._nodes[-1].outcoming_nodes.append(self)

    @abc.abstractmethod
    def apply_forward(self):
        pass

    @abc.abstractmethod
    def apply_backward(self, with_respect):
        pass

    def forward(self):
        if self.output is not None:
            return self.output
        self.output = self.apply_forward()
        return self.output

    def reset_gradients(self, nodes):
        for node1, node2 in nodes:
            node1.gradients = 0.
            node2.gradients = 0.

    def backward(self, with_respect):
        path = []
        def _build_path_to_target_variable(prev_node):
            nonlocal with_respect
            for i in prev_node.incoming_nodes:
                if not isinstance(i, Variable):
                    _build_path_to_target_variable(i)
                if (isinstance(i, Variable) and i is with_respect or isinstance(i, Node)) and (prev_node, i) not in path:
                    path.append((prev_node, i))

        _build_path_to_target_variable(self)
        self.reset_gradients(path)
        print(path)
        path = reversed(path)
        self.gradients = 1.0
        for most_recent_operation, prev_operation in path:
            most_recent_operation.apply_backward(prev_operation).data


    def __repr__(self):
        return f"<{self.__class__.__name__.capitalize()}Operation{self.counter}>"
