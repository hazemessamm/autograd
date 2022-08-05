from typing import List, TypeVar, Union
from autograd.ops_mixin import OperationsMixin

Node = TypeVar("Node", bound="Node")


class Node(OperationsMixin):
    def __init__(self, incoming_nodes: List[Node] = []):
        self.incoming_nodes = incoming_nodes
        self.outcoming_nodes = []
        self.output_node = None
        self.output = None
        self._attach_to_outcoming_nodes()

    @property
    def data(self):
        if self.output is not None:
            return self.output
        return self.forward()

    @property
    def shape(self):
        return self.data.shape

    def get_incoming_nodes(self) -> Union[List[Node], Node]:
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

    def is_last_operation(self, node):
        if len(node.outcoming_nodes) == 0:
            return True
        return False

    def compute_gradients(self, with_respect):
        path_to_target_variable = []
        latest_grad = 1.0

        def _compute_grad(node: Node):
            path_to_target_variable.append(node)
            for n in node.outcoming_nodes:
                _compute_grad(n)
        _compute_grad(with_respect)
        path_to_target_variable = list(reversed(path_to_target_variable))
        # print(path_to_target_variable)
        for most_recent_operation, prev_operation in zip(
            path_to_target_variable[:-1], path_to_target_variable[1:]
        ):
            out = most_recent_operation.backward(prev_operation).data
            latest_grad *= out
        return latest_grad

    def __repr__(self):
        return f"<{self.__class__.__name__.capitalize()} Operation>"
