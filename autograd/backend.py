from autograd.node import Node
from autograd.variable import Leaf


def reset_intermediate_gradients():
    for n in Node.instances:
        n.gradients = 0.0


def reset_leaf_gradients():
    for n in Leaf.instances:
        n.gradients = 0.0


def reset_gradients():
    reset_intermediate_gradients()
    reset_leaf_gradients()
