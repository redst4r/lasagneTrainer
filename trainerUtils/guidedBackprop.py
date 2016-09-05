"from https://github.com/Lasagne/Recipes/blob/master/examples/Saliency%20Maps%20and%20Guided%20Backpropagation.ipynb"

import warnings
import lasagne
import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
"""
not quiet clear what graident to calculate.
- currenly it takes the max-prediction an dervives wrt to the max
- also it sums over all the minibatch. that is prob WRONG

could also calculate grad wrt to certain class prob, not only the max one

"""
# TODO put in asserts that only batches of size 1 are fed. because of the SUM


def _get_relus(outputlayer):
    relu = lasagne.nonlinearities.rectify
    relu_layers = [layer for layer in lasagne.layers.get_all_layers(outputlayer)
                   if getattr(layer, 'nonlinearity', None) is relu]
    return relu_layers


def compile_saliency_function_deconv(inputlayer, outputlayer):

    relu = lasagne.nonlinearities.rectify
    relu_layers = _get_relus(outputlayer)

    # modify the Relus for deconv
    modded_relu = ZeilerBackprop(relu)
    for layer in relu_layers:
        layer.nonlinearity = modded_relu

    saliency_fn = compile_saliency_function_backprop(inputlayer, outputlayer)

    # undo the change in the relu
    for layer in relu_layers:
        layer.nonlinearity = relu

    return saliency_fn


def compile_saliency_function_guided_bp(inputlayer, outputlayer):

    """
    creates a theano function that calculates saliecny via guided backprop
    :param inputlayer:
    :param outputlayer:
    :return:
    """
    warnings.warn('this modifies the original networks ReLUs. cannot be used afterwards')
    relu = lasagne.nonlinearities.rectify
    relu_layers = _get_relus(outputlayer)
    modded_relu = GuidedBackprop(relu)  # important: only instantiate this once!
    for layer in relu_layers:
        layer.nonlinearity = modded_relu

    saliency_fn = compile_saliency_function_backprop(inputlayer, outputlayer)

    # undo the change in the relu
    for layer in relu_layers:
        layer.nonlinearity = relu

    return saliency_fn


def compile_saliency_function_backprop(input_layer, output_layer):
    """
    Compiles a function to compute the saliency maps and predicted classes
    for a given minibatch of input images.
    Uses the standard backprop

    ONE has to take care that we supply the preactivation here!
    input and ouput layers of the net are required
    :param input_layer:
    :param output_layer:
    :return:
    """

    if hasattr(input_layer, 'nonlinearity'):
        warnings.warn('usually, nonlinearity should be chopped off for saliency')

    inp = input_layer.input_var
    outp = lasagne.layers.get_output(output_layer, deterministic=True)
    max_outp = T.max(outp, axis=1)
    saliency = theano.grad(max_outp.sum(), wrt=inp)    # TODO why the sum!?! max_outp is a vector of max.probabliities
    max_class = T.argmax(outp, axis=1)
    return theano.function([inp], [saliency, max_class])


def compile_saliency_function(net):
    """
    Compiles a function to compute the saliency maps and predicted classes
    for a given minibatch of input images.
    requires a dict of layers, with specific names!

    remnant of the original ipython notebook
    """
    inp = net['input']
    outp = net['fc8']
    return compile_saliency_function_backprop(inp, outp)


def show_images(img_original, saliency, max_class, title):
    # get out the first map and class from the mini-batch
    saliency = saliency[0]
    max_class = max_class[0]
    # convert saliency from BGR to RGB, and from c01 to 01c
    saliency = saliency[::-1].transpose(1, 2, 0)
    # plot the original image and the three saliency map variants
    plt.figure(figsize=(10, 10), facecolor='w')
    plt.suptitle("Class: " + str(max_class) + ". Saliency: " + title)
    plt.subplot(2, 2, 1)
    plt.title('input')
    plt.imshow(img_original)
    plt.subplot(2, 2, 2)
    plt.title('abs. saliency')
    plt.imshow(np.abs(saliency).max(axis=-1), cmap='gray')
    plt.subplot(2, 2, 3)
    plt.title('pos. saliency')
    plt.imshow((np.maximum(0, saliency) / saliency.max()))
    plt.subplot(2, 2, 4)
    plt.title('neg. saliency')
    plt.imshow((np.maximum(0, -saliency) / -saliency.min()))
    plt.show()


class ModifiedBackprop(object):

    def __init__(self, nonlinearity):
        self.nonlinearity = nonlinearity
        self.ops = {}  # memoizes an OpFromGraph instance per tensor type

    def __call__(self, x):
        # OpFromGraph is oblique to Theano optimizations, so we need to move
        # things to GPU ourselves if needed.
        if theano.sandbox.cuda.cuda_enabled:
            maybe_to_gpu = theano.sandbox.cuda.as_cuda_ndarray_variable
        else:
            maybe_to_gpu = lambda x: x
        # We move the input to GPU if needed.
        x = maybe_to_gpu(x)
        # We note the tensor type of the input variable to the nonlinearity
        # (mainly dimensionality and dtype); we need to create a fitting Op.
        tensor_type = x.type
        # If we did not create a suitable Op yet, this is the time to do so.
        if tensor_type not in self.ops:
            # For the graph, we create an input variable of the correct type:
            inp = tensor_type()
            # We pass it through the nonlinearity (and move to GPU if needed).
            outp = maybe_to_gpu(self.nonlinearity(inp))
            # Then we fix the forward expression...
            op = theano.OpFromGraph([inp], [outp])
            # ...and replace the gradient with our own (defined in a subclass).
            op.grad = self.grad
            # Finally, we memoize the new Op
            self.ops[tensor_type] = op
        # And apply the memoized Op to the input we got.
        return self.ops[tensor_type](x)


class GuidedBackprop(ModifiedBackprop):
    def grad(self, inputs, out_grads):
        (inp,) = inputs
        (grd,) = out_grads
        dtype = inp.dtype
        return (grd * (inp > 0).astype(dtype) * (grd > 0).astype(dtype),)


class ZeilerBackprop(ModifiedBackprop):
    def grad(self, inputs, out_grads):
        (inp,) = inputs
        (grd,) = out_grads
        #return (grd * (grd > 0).astype(inp.dtype),)  # explicitly rectify
        return (self.nonlinearity(grd),)  # use the given nonlinearity




