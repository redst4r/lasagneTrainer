from pretrained.vgg19 import build_pretrained_vgg19
from nnElements import freeze
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer, get_all_layers, get_output, NINLayer
from lasagne.nonlinearities import softmax
import theano.tensor as T
from DNN_costfunctions import get_network_cost_functions
from pretrained.utils import create_channel_transformer_layers

"""
to create a slightly modified VGGnet that can handle images with a different nubmer of input channels
(other than the standard 3RGB channels)
"""


def modify_vgg19(input_var, in_channels, channel_transformers_shape, n_classes, freeze_weights):
    """
    load the VGG19 net, pretrained on Imagenet.
    however, this expects three channels, but we have N channels
    hence, put in a first layer that goes from N->3 channels. Just a linear combo of the N channels,
    i.e. a 1x1 convolution with maybe a nonlinearity

    channel_transformers_shape: tuple, each entry giving  #neurons of a layer, taking the N channels to 3 eventually,
                                e.g (3,) is a single nonlinear transform
                                (64,3) will be N channels -> 64c -> 3c
                                last tuple entry must be 3, as this is the size the vgg can handle
    """

    vgg_channels = 3  # what the pretrained net can take
    assert channel_transformers_shape[-1] == vgg_channels, "last transformer layer must have 3 filters to RGB"
    vgg_size = (224,224)

    net = build_pretrained_vgg19()

    if freeze_weights:
        print('freezing the pretrained layers')
        for l in get_all_layers(net['prob']):
            freeze(l)

    # replace the old input layer, with one that takes the 34 channels
    net['input'] = InputLayer((None, in_channels)  + vgg_size , input_var=input_var)

    # add a couple of transformations of the 34 channels -> 3 channels
    transformer_layers = create_channel_transformer_layers(net['input'], channel_transformers_shape, name_prefix='lincomb')
    for tl in transformer_layers: # add them to the dict
        net[tl.name] = tl

    # link it into the other layers
    net['conv1_1'].input_layer = transformer_layers[-1]  # conv11 was previously linked to the input, now we put the linCombo in between


    "replace the top softmax (1000 classes) by a softmax with the approropate classes"
    net['fc8'] = DenseLayer(net['fc7_dropout'], num_units=n_classes, nonlinearity=None, name='fc8')  # funny in vgg code: why not put it into a single layer with nonlin
    net['prob'] = NonlinearityLayer(net['fc8'], softmax, name='prob')

    return net


def create_vgg19_mod(in_channels, transform_shape, optimizer, optimizer_params, n_classes, freeze_weights):
    """
    creates a modified VGG, and the corresponding training/validation functions
    :param in_channels: number of input channels
    :param transform_shape: how to transform the intput channels into RGB (as expected by VGG), see doc. of modify_vgg19
    :param optimizer: usually 'adam'
    :param optimizer_params: parameters passed to the optimizer, e.g. {'learning_rate': 0.1}
    :param n_classes: number of classes to classify
    :param freeze_weights: boolean, if True, the weights of the VGG are frozen, and only the transformer weights are learned in training
    :return:
    """

    'theano variables for the neural net'
    input_var = T.tensor4('inputs')    # symbol for the input images
    target_var = T.ivector('targets')  # symbol for the labels

    net_dict = modify_vgg19(input_var, in_channels, transform_shape, n_classes, freeze_weights)

    train_fn, val_fn = get_network_cost_functions(net_dict['prob'], input_var, target_var, optimizer, optimizer_params)

    return net_dict['prob'], train_fn, val_fn
