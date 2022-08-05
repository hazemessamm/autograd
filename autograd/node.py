from typing import List, TypeVar, Union
from autograd.ops_mixin import OperationsMixin
import numpy as np

Node = TypeVar("Node", bound="Node")


class Node(OperationsMixin):
    def __init__(self, incoming_nodes: List[Node] = []):
        # incoming operations or variables
        self.incoming_nodes = incoming_nodes
        # outcoming operations or variables
        self.outcoming_nodes = []
        # if a node has nested nodes in it 
        # then the last nested node should be stored here
        self.output_node = None
        # for caching outputs
        self.output = None
        # for connecting nodes
        self._attach_to_outcoming_nodes()
        # caching gradients
        self.gradients = None

    @property
    def data(self):
        if self.output is not None:
            return self.output
        return self.forward()

    @property
    def shape(self):
        return self.data.shape

    def get_incoming_nodes(self) -> Union[List[Node], Node]:
        '''Returns incoming nodes and also checks 
        if an incoming node has nested nodes 
        so it can return the last nested node'''

        if len(self.incoming_nodes) > 1:
            return self.incoming_nodes

        incoming_node = self.incoming_nodes[0]
        if len(self.incoming_nodes) == 1:
            if incoming_node.output_node is not None:
                return incoming_node.output_node
            else:
                return incoming_node
        else:
            return self.incoming_nodes

    def _attach_to_outcoming_nodes(self):
        for node in self.incoming_nodes:
            node.outcoming_nodes.append(self)

        for n in self.incoming_nodes:
            if isinstance(n, Node) and n.output_node is not None:
                n.output_node.outcoming_nodes.append(self)

    def forward(self):
        raise NotImplementedError

    def backward(self, with_respect):
        raise NotImplementedError

    def compute_gradients(self, with_respect, save_gradients=True):
        path = []
        latest_grad = 1.0

        def _build_path_to_target_variable(node: Node):
            path.append(node)
            for n in node.outcoming_nodes:
                _build_path_to_target_variable(n)

        _build_path_to_target_variable(with_respect)
        path = list(reversed(path))
        for most_recent_operation, prev_operation in zip(path[:-1], path[1:]):
            out = most_recent_operation.backward(prev_operation).data
            # sum the latest_grad if the upcoming grad is a scalar variable
            if len(out.shape) < 1 or out.shape[0] == 1 and latest_grad.shape[0] > 1:
                latest_grad = np.sum(latest_grad)
            latest_grad *= out
        
        if save_gradients:
            with_respect.gradients = latest_grad
        return latest_grad


    def __repr__(self):
        return f"<{self.__class__.__name__.capitalize()} Operation>"
