from pretrained.inception_v3 import build_pretrained_inception_v3
from nnElements import freeze
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer, get_all_layers, get_output, NINLayer
from lasagne.nonlinearities import softmax
import theano.tensor as T
from DNN_costfunctions import get_network_cost_functions
from pretrained.utils import create_channel_transformer_layers
import warnings


def modify_inception3(input_var, in_channels, channel_transformers_shape, n_classes):
    """
    load the inception_v3 net, pretrained on Imagenet
    however, this expects three channels, but we might have a differnet number N of channels
    hence, put in a first layer that goes from N->3 channels. Just a linear combo of the N channels, i.e. a 1x1 convolution with maybe a nonlinearity

    channel_transformers_shape: tuple, each entry giving  #neurons of a layer, taking the N channels to 3 eventually, e.g (3,) is a single nonlinear transform
                                (64,3) will be Nc -> 64c -> 3c
                                last tuple entry must be 3, as this is the size the inception can handle
    """

    incept_channels = 3  # what the pretrained net can take
    assert channel_transformers_shape[-1] == incept_channels, "last transformer layer must have 3 filters to RGB"
    incept_size = (299,299)

    net = build_pretrained_inception_v3()
    warnings.warn("fix all pretrained weights and only optimize the linCombo layers?!")
    for l in get_all_layers(net['prob']):
        freeze(l)

    # replace the old input layer, with one that takes the N channels
    net['input'] = InputLayer((None, in_channels)  + incept_size , input_var=input_var)

    # add a couple of transformations of the N channels -> 3 channels
    transformer_layers = create_channel_transformer_layers(net['input'], channel_transformers_shape, name_prefix='lincomb')
    for tl in transformer_layers: # add them to the dict
        net[tl.name] = tl

    # link it into the other layers
    net['conv_1'].input_layer = transformer_layers[-1]  # conv1 was previously linked to the input, now we put the linCombo in between


    "replace the top softmax (1000 classes) by a softmax with the approropate classes"
    # the last layer before the softmax:
    hidden_layer = net['pool3']

    net['softmax'] = DenseLayer(hidden_layer, num_units=n_classes, nonlinearity=softmax, name='softmax')

    return net


def create_inception3_mod(in_channels, transform_shape, optimizer, optimizer_params, n_classes):
    """
    creates the modified inception net and train/val_functions
    :param in_channels:  # input channels
    :param transform_shape:  # tuple, specifying the sequence of dimension reductions to get to three colors
    :param optimizer:
    :param optimizer_params:
    :param n_classes: # of output classes
    :return:
    """
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')  # symbol for the labels

    net_dict = modify_inception3(input_var, in_channels, transform_shape, n_classes)

    train_fn, val_fn = get_network_cost_functions(net_dict['softmax'], input_var, target_var, optimizer, optimizer_params)

    return net_dict['softmax'], train_fn, val_fn
