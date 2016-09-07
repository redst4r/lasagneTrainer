import numpy as np
from lasagne.layers import get_output, InputLayer, DenseLayer, ReshapeLayer
from lasagne.nonlinearities import softmax
import theano
import lasagne.objectives
import theano.tensor as T
from nnElements import _dense_and_dropout, _conv_and_maxpool
from NetworkTrainer import NetworkTrainer
from batchgenerators import iterate_batches_from_disk, random_crops_iterator


def get_network_cost_functions(network, input_var, target_var, optimizer, optimizer_params):
    """
    given a DNN, create the cost function, gradients etc used to train the network
    simple cross entropy loss on the last layer

    :param network: last layer of the DNN
    :param input_var: theano variable representing the input (T.tensor4)
    :param target_var: theano variable representing the labels (T.ivector)
    :return: training and validation theano functions
    """
    # training loss
    prediction = get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    train_prediction = lasagne.layers.get_output(network, deterministic=False)
    train_acc = T.mean(T.eq(T.argmax(train_prediction, axis=1), target_var), dtype=theano.config.floatX)


    # validation loss
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

    params = lasagne.layers.get_all_params(network, trainable=True)

    if optimizer == 'adam':
        updates = lasagne.updates.adam(loss, params, **optimizer_params)
    elif optimizer == 'nesterov':
        if optimizer_params is None:
            print('warning: using default parameters for nesterov')
            optimizer_params = {'learning_rate': 0.05, 'momentum': 0.9}
        updates = lasagne.updates.nesterov_momentum(loss, params, **optimizer_params)
    else:
        raise Exception('Optimizer not recognized: %s' % optimizer)
    train_fn = theano.function([input_var, target_var], [loss, train_acc], updates=updates)
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    return train_fn, val_fn