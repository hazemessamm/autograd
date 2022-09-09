AUTOMATIC_GRADIENT_RESET = True
FLUSH_OLD_OPERATIONS = True



def enable_reset_gradients(state: bool):
    if not isinstance(state, bool):
        raise ValueError(f'`state` should be an instance of `bool`. Recieved: {type(state)}')
    global AUTOMATIC_GRADIENT_RESET
    AUTOMATIC_GRADIENT_RESET = state


def reset_gradient_enabled():
    global AUTOMATIC_GRADIENT_RESET
    return AUTOMATIC_GRADIENT_RESET


def reset_intermediate_gradients():
    from autograd.node import Node
    for n in Node.instances:
        n.gradients = 0.


def reset_leaf_gradients():
    from autograd.variable import Leaf
    for n in Leaf.instances:
        n.gradients = 0.


def reset_gradients():
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