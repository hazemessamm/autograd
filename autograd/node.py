import abc
import threading
import weakref
from typing import List, TypeVar, Union

from autograd import backend
from autograd.exceptions import NoPathFoundError
from autograd.ops_mixin import OperationsMixin
from autograd.variable import Leaf, Variable

Node = TypeVar("Node", bound="Node")


class Node(abc.ABC, OperationsMixin):
    instances = weakref.WeakSet()
    num_instances = 0

    def __init__(self, incoming_nodes: List[Node] = [], name: str = None):
        # incoming operations or variables
        self.incoming_nodes = incoming_nodes
        # outcoming operations or variables
        self.outcoming_nodes = []
        # for connecting nodes
        # caching gradients
        self.gradients = 0.0
        self.nested_nodes = None
        self.nested = False
        Node.instances.add(self)
        self.cached_graphs = {}
        self._attach_to_outcoming_nodes()

        with threading.Lock():
            Node.num_instances += 1
            self.counter = Node.num_instances
        
        if name is None:
            self.name = f'<{self.__class__.__name__.capitalize()}Operation{self.counter}>'
        else:
            self.name = name

    def __setattr__(self, __name: str, __value: Node) -> None:
        if isinstance(__value, Node):
            if self.nested_nodes is None:
                self.nested_nodes = []
                self.nested = True
            self.nested_nodes.append(__value)
        return super().__setattr__(__name, __value)

    @property
    def data(self):
        return self.forward()

    @property
    def shape(self):
        return self.data.shape

    def get_output_node(self):
        return self.nested_nodes[-1] if self.nested else None

    def get_incoming_nodes(self) -> Union[List[Node], Node]:
        """Returns incoming nodes and also checks
        if an incoming node has nested nodes
        so it can return the last nested node"""

        if len(self.incoming_nodes) > 1:
            return self.incoming_nodes

        incoming_node = self.incoming_nodes[0]
        if not len(self.incoming_nodes) == 1:
            return self.incoming_nodes

        if isinstance(incoming_node, Node) and incoming_node.nested:
            return incoming_node.nested_nodes[-1]
        else:
            return incoming_node


    def _attach_to_outcoming_nodes(self):
        for node in self.incoming_nodes:
            node.outcoming_nodes.append(self)
            
            if isinstance(node, Node):
                Node.instances.add(node)

    @abc.abstractmethod
    def apply_forward(self):
        pass

    @abc.abstractmethod
    def apply_backward(self, with_respect):
        pass

    def forward(self):
        return self.apply_forward()

    def reset_gradients(self, nodes):
        for node1, node2 in nodes:
            node1.gradients = 0.0
            node2.gradients = 0.0

    def _build_graph_to_target_variable(self, with_respect):
        path = []
        not_variable_error_flag = True

        def traverse(node):
            nonlocal path, with_respect, not_variable_error_flag
            for i in node.incoming_nodes:
                if not isinstance(i, Leaf):
                    traverse(i)
                if ((isinstance(i, Leaf) and i is with_respect) or isinstance(i, Node)) and (node, i) not in path:
                    if i is with_respect:
                        not_variable_error_flag = False
                    path.append((node, i))

        traverse(self)
        if not_variable_error_flag:
            raise NoPathFoundError(f"Cannot create a graph for variable {with_respect}")
        return path

    def backward(self, with_respect):
        if isinstance(with_respect, (tuple, list)):
            self._multi_variable_backward(with_respect)
        else:
            self._single_variable_backward(with_respect)
    
    def _single_variable_backward(self, with_respect):
        path = self._build_graph_to_target_variable(with_respect)
        self.cached_graphs[with_respect] = path

        if backend.reset_gradient_enabled():
            self.reset_gradients(path)

        self.gradients = 1.0
        for most_recent_operation, prev_operation in reversed(path):
            most_recent_operation.apply_backward(prev_operation)

    def _multi_variable_backward(self, variables):
        for var in backend.flatten(variables):
            self._single_variable_backward(var)

    def __repr__(self):
        return self.name
