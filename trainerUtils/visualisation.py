from lasagne.layers import get_output
from lasagne.layers import get_output_shape
from lasagne.objectives import binary_crossentropy
import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T
import progressbar
import lasagne
from plottingUtils import tile_raster_images

def viz_kernels_in_a_row(c1, c2):
    from scipy.signal import convolve2d
    from functools import reduce
    """
    vizualize the consecutive action of two kernels in a row
    :param c1: first convolutional layer
    :param c2: second convolutional layer
    :return:
    """

    """
    its complicated:
    the first layer  has dimension N x 1 x 3 x 3  (N filters, 1 channel)
    the second layer has dimension M x N x 3 x 3 (M filters, N channel)

    filter n=1 from the first layer is (as intermediate) convolved with filter m=1 in the second layer
    only in 1 x n x 3 x 3
    """

    w1 = c1.W.get_value()
    w2 = c2.W.get_value()

    assert w1.shape[0] == w2.shape[1], "first layer #filter must match 2nd layer #channels"

    combinedFilter = []
    # iterate over the sencond layers filters (which are 3d) and convolve with the "single" 3d filter from the first layer
    for i in range(w2.shape[0]):
        # stupid we cant do 2d convolution on 3D arrays (conv along the first two)
        tmp = [convolve2d(w1[j,0,:,:], w2[i, j,:,:]) for j in range(w1.shape[0])]   # w1[j,0] since the second dim is channgels
        # tmp is now  Nx5x5
        theFilter = reduce(lambda x,y: x+y, tmp[:2]) # sum across all those N intermediates to get the final filter
        combinedFilter.append(theFilter)


"""
-----------------------------------------------------------------------------------------------------------
COPIED FROM nolearn.lasagne.visualize.py
modified to work without nolearn.NeuralNet
-----------------------------------------------------------------------------------------------------------
"""


def occlusion_heatmap(last_layer, x, target, square_length=7, batchsize=10):
    """An occlusion test that checks an image for its critical parts.

    In this function, a square part of the image is occluded (i.e. set
    to 0) and then the net is tested for its propensity to predict the
    correct label. One should expect that this propensity shrinks of
    critical parts of the image are occluded. If not, this indicates
    overfitting.

    Depending on the depth of the net and the size of the image, this
    function may take awhile to finish, since one prediction for each
    pixel of the image is made.

    Currently, all color channels are occluded at the same time. Also,
    this does not really work if images are randomly distorted by the
    batch iterator.

    See paper: Zeiler, Fergus 2013

    Parameters
    ----------
    last_layer : lasagne.layer instance
      The output layer of the Neural net

    x : np.array
      The input data, should be of shape (1, c, x, y). Only makes
      sense with image data.

    target : int
      The true value of the image. If the net makes several
      predictions, say 10 classes, this indicates which one to look
      at.

    square_length : int (default=7)
      The length of the side of the square that occludes the image.
      Must be an odd number.

    Results
    -------
    heat_array : np.array (with same size as image)
      An 2D np.array that at each point (i, j) contains the predicted
      probability of the correct class if the image is occluded by a
      square with center (i, j).

    """
    if (x.ndim != 4) or x.shape[0] != 1:
        raise ValueError("This function requires the input data to be of "
                         "shape (1, c, x, y), instead got {}".format(x.shape))
    if square_length % 2 == 0:
        raise ValueError("Square length has to be an odd number, instead "
                         "got {}.".format(square_length))

    num_classes = get_output_shape(last_layer)[1]
    img = x[0].copy()
    bs, col, s0, s1 = x.shape

    heat_array = np.zeros((s0, s1))
    pad = square_length // 2 + 1
    x_occluded = np.zeros((s1, col, s0, s1), dtype=img.dtype)
    probs = np.zeros((s0, s1, num_classes))

    def predict(layer, inputImgs, batchsize):
        scores = []
        for start in np.arange(0, inputImgs.shape[0], batchsize):
            thebatch = inputImgs[start:(start + batchsize)]
            probs = get_output(layer, thebatch).eval()
            assert probs.shape == (len(thebatch), num_classes)
            scores.append(probs)
        return np.vstack(scores)

    # generate occluded images
    bar = progressbar.ProgressBar()
    for i in bar(range(s0)):
        # batch s1 occluded images for faster prediction
        for j in range(s1):
            x_pad = np.pad(img, ((0, 0), (pad, pad), (pad, pad)), 'constant')
            x_pad[:, i:i + square_length, j:j + square_length] = 0.
            x_occluded[j] = x_pad[:, pad:-pad, pad:-pad]

        y_proba = predict(last_layer, x_occluded, batchsize)
        probs[i] = y_proba.reshape(s1, num_classes)

    # from predicted probabilities, pick only those of target class
    for i in range(s0):
        for j in range(s1):
            heat_array[i, j] = probs[i, j, target]
    return heat_array, probs


def plot_saliency_map(sMap, X):

    # salmaps usually come as W x H x channels, but images as channel x  W x H
    sMap = sMap.transpose([2,0,1])

    assert len(sMap.shape) == 3
    channels, W, H = X.shape
    samples = 1
    assert X.shape == sMap.shape

    figsize = 10
    figsize = (figsize, samples * figsize / 3)
    figs, axes = plt.subplots(samples*channels, 3, figsize=figsize)

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')


    for n in range(channels):
        x_channel = X[n,:,:]
        smap_channel = sMap[n,:,:]
        ax = axes[n]

        ix_sp = n*3
        ax[0].imshow(x_channel, interpolation='nearest', cmap='gray')
        ax[0].set_title('image')
        ax[1].imshow(smap_channel, interpolation='nearest', cmap='Reds')  #  vmin=0, vmax=1
        ax[1].set_title('critical parts')
        ax[2].imshow(x_channel, interpolation='nearest', cmap='gray')
        ax[2].imshow(smap_channel, interpolation='nearest', cmap='Reds',
                     alpha=0.6)
        ax[2].set_title('super-imposed')




def plot_heatmap(heatmap, X, figsize):
    if (X.ndim != 4):
        raise ValueError("This function requires the input data to be of "
                         "shape (b, c, x, y), instead got {}".format(X.shape))

    num_images = X.shape[0]
    if figsize[1] is None:
        figsize = (figsize[0], num_images * figsize[0] / 3)
    figs, axes = plt.subplots(num_images, 3, figsize=figsize)

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')

    for n in range(num_images):
        heat_img = heatmap.sum(-1)  # summing over color channels

        ax = axes if num_images == 1 else axes[n]
        img = X[n, :, :, :].mean(0)
        ax[0].imshow(-img, interpolation='nearest', cmap='gray')
        ax[0].set_title('image')
        ax[1].imshow(-heat_img, interpolation='nearest', cmap='Reds')  #  vmin=0, vmax=1
        ax[1].set_title('critical parts')
        ax[2].imshow(-img, interpolation='nearest', cmap='gray')
        ax[2].imshow(-heat_img, interpolation='nearest', cmap='Reds',
                     alpha=0.6)
        ax[2].set_title('super-imposed')
    return plt


def saliency_map(input, output, pred, X):
    score = -binary_crossentropy(output[:, pred], np.array([1]).astype('float32')).sum()  # no idea why this conversion has to be done to float, but in32 results in floats returning
    return np.abs(T.grad(score, input).eval({input: X}))  # WHY ABS?!


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

    output_before_nonlin = _get_output_before_nonlinearity(outputlayer, deterministic=deterministic)

    # TODO test output == outputlayer.nonlinearity(outputlayer.b + input_into_last.dot(outputlayer.W))
    # but for determinsitic
    output = get_output(outputlayer , deterministic=deterministic)
    pred = output.eval({input: X}).argmax(axis=1).astype('int32')  # that just picks out the most likely class here

    if chop_nonlin:
        sali = saliency_map(input, output, pred, X)
    else:
        sali = saliency_map(input, output_before_nonlin, pred, X)
    return sali[0].transpose(1, 2, 0).squeeze()

#
# def plot_saliency(net, X, figsize=(9, None)):
#     return _plot_heat_map(
#         net, X, figsize, lambda net, X, n: -saliency_map_net(net, X))


def synthesize_image(input_layer,output_layer, inputshape, which_class, gradient_steps,gradient_stepsize, LAM, chopNonlin=True, I0=None):
    """
    does gradient ascend in image space to maximize a certain class score, hence producing an image
    that maximizes a class
    :param input_layer:
    :param output_layer:
    :param inputshape:
    :param gradient_steps:
    :param gradient_stepsize:
    :param chopNonlin: maximize the class score before or after the nonlinearity: =True-> maximize the unnormalized score
    :param I0: optinally, put in an image from which we start the optimization. Could be a natural image in whihc we want to enhance the features of the clas
           If None, random initialization will be made
    :return:
    """
    from lasagne.regularization import l2

    assert inputshape[0]==1
    input_var = input_layer.input_var

    if chopNonlin:
        before_non =_get_output_before_nonlinearity(output_layer, deterministic=True)
        classNeuron = before_non[0, which_class]
    else:
        classNeuron = get_output(output_layer, deterministic=True)[0,which_class]  # the zero is needed to get a scalar gradient (otherwise we would a single number per image)! we want a single image anyways

    regularized_score = classNeuron - LAM * l2(input_var)  # mind the SIGN: we MAXIMIZE class_prob, hence l2 must be subtracted
    theGrad = T.grad(regularized_score, input_var)

    # gradient_fn = theano.function([input_var], theGrad)
    score_fn = theano.function([input_var], regularized_score)

    # I = np.zeros(inputshape,dtype='float32')

    # starting point of gradient ascend:
    if I0 is None:
        I = np.random.normal(0,1, inputshape).astype('float32')
    else:
        I = I0
        assert I.shape == inputshape

    I_progress = []
    score_progress = []
    bar = progressbar.ProgressBar()
    for i in bar(range(gradient_steps)):
        if i % 10 == 0:
            I_progress.append(I.copy())
        gr = theGrad.eval({input_var: I})
        I += gr * gradient_stepsize
        the_score = score_fn(I)
        score_progress.append(the_score)
        # print("%d\t%.3f" % (i, the_score))

    plt.figure();plt.plot(score_progress)
    return I_progress, score_progress
    # tile_raster_images(np.concatenate(I_progress), (28, 28), (10, 10), tile_spacing=(1, 1),scale_rows_to_unit_interval=True)

