from logging import warn


AUTOMATIC_GRADIENT_RESET = True
FLUSH_OLD_OPERATIONS = True



def enable_reset_gradients(state: bool):
    if not isinstance(state, bool):
        raise ValueError(f'`state` should be an instance of `bool`. Recieved: {type(state)}')
    if not state:
        warn('`enable_reset_gradients` still not ready')
    global AUTOMATIC_GRADIENT_RESET
    AUTOMATIC_GRADIENT_RESET = state


def reset_gradient_enabled():
    global AUTOMATIC_GRADIENT_RESET
    return AUTOMATIC_GRADIENT_RESET


def reset_intermediate_gradients():
    warn('`reset_intermediate_gradients` still not ready')
    from autograd.node import Node
    for n in Node.instances.data:
        n().gradients = 0.


def reset_leaf_gradients():
    warn('`reset_leaf_gradients` still not ready')
    from autograd.variable import Leaf
    for n in Leaf.instances.data:
        n().gradients = 0.


def reset_gradients():
    warn('`reset_gradients` still not ready')
    reset_intermediate_gradients()
    reset_leaf_gradients()


def flatten(lists):
    result = []
    def _flatten(l):
        for x in l:
            if isinstance(x, list):
                _flatten(x)
            else:
                result.append(x)
    _flatten(lists)
    return result


def reset_weights_graph(weights):
    for w in flatten(weights):
        w.outcoming_nodes = []


def sgd_update(weights, lr=0.01):
    for w in flatten(weights):
        w._data += -lr * w.gradients