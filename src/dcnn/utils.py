from collections import OrderedDict

import numpy as np

import theano
import theano.tensor as T
import lasagne.updates


# Adapted from Lasagne

def adagrad(loss_or_grads, params, learning_rate=1.0, epsilon=1e-6):
    grads = lasagne.updates.get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()
    accus = []

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
        accu_new = accu + grad ** 2
        updates[accu] = accu_new
        accus.append((accu, value.shape))
        updates[param] = param - (learning_rate * grad /
                                  T.sqrt(accu_new + epsilon))

    return updates, accus


def reset_grads(accus):
    for accu in accus:
        accu[0].set_value(np.zeros(accu[1], dtype=accu[0].dtype))
