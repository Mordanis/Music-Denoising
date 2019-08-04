import keras.backend as K
from keras.layers import Layer
import tensorflow as tf

@tf.custom_gradient
def flip_gradient(x):
    def grad(dy):
        print(dy)
        return -1. * dy
    print('x is {}'.format(x))
    print('return val is {}'.format(1. * x))
    return (1. * x, grad)

class grad_flipper(Layer):

    def __init__(self, _=None, **kwargs):
        super(grad_flipper, self).__init__(**kwargs)
        return None

    def build(self, input_shape):
        super(grad_flipper, self).build(input_shape)
        return None

    def call(self, inputs):
        print(inputs)
        outputs = flip_gradient(inputs)
        print('outputs are {}'.format(outputs))
        return outputs

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        return output_shape
    

