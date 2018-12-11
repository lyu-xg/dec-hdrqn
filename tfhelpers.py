import tensorflow as tf
from functools import reduce
from params import CNN_Feature_Extractors

def conv2d(input, output_dimension, kernel, stride,
           format='NCHW', padding='SAME', name='conv2d',
           activation=tf.nn.relu,  
           initializer=tf.contrib.layers.xavier_initializer()):
    '''
    Add conv2d layer to network.
    :param tensor input: Input to be processed by layer.
    :param tensor output_dimension: Dimensions of layer output.
    :param list kernel: kernel dimensions
    :param list stride: stride values
    :param activation: TF activation function or set to None
    :param format: NHWC/NCHW (Batch, height, width and channels) 
    :param string padding: Type of padding to be used
    :param string name: Layer name
    :param initializer: Variable initializer
    :return tensor output
    '''
    with tf.variable_scope(name) as scope:
        if format == 'NCHW':
            stride = [1, 1, stride[0], stride[1]]
            kernel_shape = [kernel[0], 
                            kernel[1], 
                            input.get_shape()[1], 
                            output_dimension]
        elif format == 'NHWC':
            stride = [1, stride[0], stride[1], 1]
            kernel_shape= [kernel[0], 
                           kernel[1], 
                           input.get_shape()[-1], 
                           output_dimension]
        # Weights are innitialised based upon kernel size and output dimensions:
        w = tf.get_variable('w', kernel_shape, tf.float32, initializer=initializer)
        b = tf.get_variable('biases', [output_dimension], initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(input, w, stride, padding, data_format=format)
        output = tf.nn.bias_add(conv, b, format)
    # Output is returned
    return activation(output) if activation != None else output


def convLayers(inputs):
    '''
    Conv layers that can be used for feature extraction
    :param vector: inputs
    :return: Add conv features layers to tf graph
    '''
    conv_config = CNN_Feature_Extractors()
    inputs = tf.div(inputs, conv_config.max_in)
    layer = 0
    for (o, k, s) in zip(conv_config.outdim, conv_config.kernels, conv_config.stride):
        inputs = conv2d(inputs, o, [k, k], [s, s], name='conv' + str(layer))
        layer += 1                
    shape = inputs.get_shape().as_list()
    inputs = tf.reshape(inputs, [-1, reduce(lambda x, y: x * y, shape[1:])])
    return fully_connected(inputs, conv_config.fc, init_val=0.1)

def fully_connected(inputs, n_unit, activation=tf.nn.relu, init_val=0.01):
    return tf.contrib.layers.fully_connected(
        inputs, n_unit, activation_fn= activation,
        # weights_initializer=tf.initializers.xavier_initializer(),
        biases_initializer=tf.constant_initializer(value=init_val)
    )