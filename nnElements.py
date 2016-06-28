"""
functions to construct some very common patterns in neural networks
e.g. conv/maxpool pairs

"""
from lasagne.layers import get_output, InputLayer, DenseLayer, Upscale2DLayer, ReshapeLayer, GaussianNoiseLayer, DropoutLayer, SliceLayer, ConcatLayer,get_all_layers
import warnings
from lasagne.layers.normalization import batch_norm
from lasagne.nonlinearities import rectify

def _my_batch_norm(in_layer):
    """
    my modified version of batchnorm: for ReLUs neither scaling nor bias is needed
    bias is already removed by lasagne.layers.batchnorm, but the scalign is still present.
    one can enforce fixed sclaing by passing 'gamma': None to the Batchnorm Layer
    see https://github.com/Lasagne/Lasagne/issues/635
    :param layer:
    :return:
    """

    if in_layer.nonlinearity.__name__ == rectify.__name__:
        kw = {'gamma': None}
    else:
        kw = {}

    return batch_norm(in_layer, **kw)


def _dense_and_dropout(inputlayer, dense_params, dropout_params, do_batchnorm=False):
    "convenience for creating a series of dense+dropout layers"
    dense = DenseLayer(inputlayer, **dense_params)
    if do_batchnorm:
        dense = _my_batch_norm(dense)

    drop = DropoutLayer(dense, **dropout_params)
    return drop


def _conv_and_maxpool(input_layer, conv_params, maxpool_params, do_batchnorm=False):
    "convenience for creating a series of conv+maxpool layers"

    conv1_layer = _create_conv2D_layer(input_layer, DNN_mode='cuDNN', do_batchnorm=do_batchnorm, **conv_params)
    max1_layer  = _create_maxpool2D_layer(conv1_layer, DNN_mode='cuDNN', **maxpool_params)
    return max1_layer


def _conv_conv_pool(input_layer, conv1_params, conv2_params, maxpool_params, do_batchnorm=False):
    "2convolitions followed by maxpooling. very common: 3x3 -> 3x3 -> pool"
    conv1_layer   = _create_conv2D_layer(input_layer, DNN_mode='cuDNN', do_batchnorm=do_batchnorm ,**conv1_params)
    conv2_layer   = _create_conv2D_layer(conv1_layer, DNN_mode='cuDNN', do_batchnorm=do_batchnorm, **conv2_params)
    max1_layer = _create_maxpool2D_layer(conv2_layer, DNN_mode='cuDNN' , **maxpool_params)
    return max1_layer


def _create_conv2D_layer(input_layer, DNN_mode, do_batchnorm, **conv1_params):
    "creates a conv layer using either cuDNN, cuda_convnet or standard lasagne"
    if DNN_mode == 'cuDNN':
        from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayerFast
    elif DNN_mode == 'cuda_convnet':
        from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayerFast
    else:
        from lasagne.layers import Conv2DLayer as Conv2DLayerFast

    CL = Conv2DLayerFast(input_layer, **conv1_params)
    if do_batchnorm:
        CL = _my_batch_norm(CL)
    return CL


def _create_maxpool2D_layer(input_layer, DNN_mode, **maxpool_params):
    "creates a maxpool layer using either cuDNN, cuda_convnet or standard lasagne"
    if DNN_mode == 'cuDNN':
        from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPool2DLayerFast
    elif DNN_mode == 'cuda_convnet':
        from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayerFast
    else:
        from lasagne.layers import MaxPool2DLayer as MaxPool2DLayerFast

    return MaxPool2DLayerFast(input_layer, **maxpool_params)
