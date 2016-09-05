from lasagne.layers import get_output
from lasagne.objectives import binary_crossentropy
import numpy as np
import theano
import theano.tensor as T
import lasagne


def saliency_map(input, output, pred):
    # todo cross entropy doesnt make sense if chopping the nonlinearity
    score = -binary_crossentropy(output[:, pred], np.array([1]).astype('float32')).sum()  # no idea why this conversion has to be done to float, but in32 results in floats returning
    return abs(T.grad(score, input))  # WHY ABS?!


def _get_output_before_nonlinearity(layer, deterministic):
    "grabs the neurons activations before applying the nonlinearity"
    assert isinstance(layer, lasagne.layers.DenseLayer)

    input_into_last = get_output(layer.input_layer, deterministic=deterministic)
    output_before_nonlin = layer.b + input_into_last.dot(layer.W)
    return output_before_nonlin


def saliency_map_net(inputlayer, outputlayer, X, chop_nonlin=True, deterministic=True):
    """
    :param inputlayer:
    :param outputlayer:
    :param X:
    :param chop_nonlin: whether to calculate the derivative of the class score before or after the softmax
    :return:
    """
    input = inputlayer.input_var

    # TODO test output == outputlayer.nonlinearity(outputlayer.b + input_into_last.dot(outputlayer.W))
    # but for determinsitic
    "get the predicted class labels"
    output = get_output(outputlayer , deterministic=deterministic)
    pred = output.eval({input: X}).argmax(axis=1).astype('int32')  # that just picks out the most likely class here

    if chop_nonlin:  # if we want to chop of the nonlinearity, modify the output
        output = _get_output_before_nonlinearity(outputlayer, deterministic=deterministic)

    # create a theano function out of it
    sal_fn = theano.function([input],  saliency_map(input, output, pred))
    sali = sal_fn(X).transpose(0, 2, 3, 1)
    return sali
