from typing import TypeVar, List

Node = TypeVar('Node', bound='Node')

class Node:
    def __init__(self, incoming_nodes: List[Node] = []):
        self.incoming_nodes = incoming_nodes
        self.outcoming_nodes = []
        self._output = None
        self._attach_to_outcoming_nodes()

    @property
    def data(self):
        return self.forward()

    @property
    def output(self):
        return self.forward()

    def _attach_to_outcoming_nodes(self):
        for node in self.incoming_nodes:
            node.outcoming_nodes.append(self)

    def forward(self):
        raise NotImplementedError

    def backward(self, variable):
        raise NotImplementedError

    def is_leaf(self, node):
        if len(node.outcoming_nodes) == 0:
            return True
        return False

    def compute_gradients(self, with_respect):
        path_to_target_variable = []
        latest_grad = 1.
        def _compute_grad(node: Node):
            if self.is_leaf(node):
                return
            path_to_target_variable.append(node)
            for n in node.outcoming_nodes:
                _compute_grad(n)
        _compute_grad(with_respect)
        path_to_target_variable.append(self)
        path_to_target_variable = list(reversed(path_to_target_variable))
        for most_recent_operation, prev_operation in zip(path_to_target_variable[:-1], path_to_target_variable[1:]):
            out = most_recent_operation.backward(prev_operation).data
            latest_grad *= out
        return latest_grad
    
    def __repr__(self):
        return f"<{self.__class__.__name__.capitalize()} Operation>"