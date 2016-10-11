from nnElements import _create_conv2D_layer, freeze
import lasagne


def create_channel_transformer_layers(input_layer, channel_transformers_shape, name_prefix, nonlinearity='ReLU'):
    """
    creates a series of channel transformations (no real covolutions but just linear combinations of the channels some nonlin act)

    :param input_layer:
    :param channel_transformers_shape:
           tuple, each entry giving  #neurons of a layer, taking the 34 channels to 3 eventually, e.g (3,) is a single nonlinear transform
           (64,3) will be 34c -> 64c -> 3c
           last tuple entry must be 3, as this is the size the vgg can handle
    :param nonlinearity: string, determining the nonlineartiy used, 'linear' or 'ReLU';
           all layers constructed ahve the same nonlinearity

    :return: list of the layers created
    """
    if nonlinearity== 'ReLU':
        nonlin = lasagne.nonlinearities.rectify
    elif nonlinearity == 'linear':
        nonlin = lasagne.nonlinearities.linear
    else:
        raise ValueError('unknown nonlinearity')

    last_in = input_layer
    "insert the linear combo layer 34->3 in the beginning"
    the_cascade =[]
    for i,n_neurons in enumerate(channel_transformers_shape):
        conv_params = {'name': '%s_%d'%(name_prefix,i) ,
                       'num_filters': n_neurons,
                       'filter_size': 1,
                       'pad': 'valid',
                       'nonlinearity': nonlin}
        comboLayer = _create_conv2D_layer(last_in, do_batchnorm=False, **conv_params)
        the_cascade.append(comboLayer)
        last_in = comboLayer

    return the_cascade

