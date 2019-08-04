import keras.backend as K 
from keras.layers import Conv1D, Dense, Activation, BatchNormalization, GaussianNoise, Input
from keras.layers import concatenate, UpSampling1D, Subtract, Flatten
from keras.models import Model
from keras.optimizers import Nadam
from keras import callbacks
from keras.utils.vis_utils import plot_model
import grad_flipper as gf
## AutoEncoder Model

def AutoEncoder (input_shape):
    in_layer = Input(input_shape)
    block_input = GaussianNoise(0.0)(in_layer)
    noisy_input = block_input
    filts = 4
    block = block_input
    nodes = []
    for j in range(2):
        block_input = block
        block_output = []
        for i in range(3):
            block = Conv1D(filters=filts, kernel_size=9, padding='same')(block_input)
            block = Activation('elu')(block)
            block = BatchNormalization()(block)
            block_output.append(block)

            block2 = Conv1D(filters=filts, kernel_size=9, padding='same', dilation_rate=4)(block_input)
            block2 = Activation('elu')(block2)
            block2 = BatchNormalization()(block2)
            block_output.append(block2)


            block3 = Conv1D(filters=filts, kernel_size=9, padding='same', dilation_rate=16)(block_input)
            block3 = Activation('elu')(block3)
            block3 = BatchNormalization()(block3)
            block_output.append(block3)

            block_input = concatenate([block_input, block, block2, block3])

        block = concatenate(block_output)
        nodes.append(block)
        block = Conv1D(filters=filts * 3, kernel_size=9, padding='same', strides=4)(block)
        block = Activation('elu')(block)
        filts *= 2
        block_input = BatchNormalization()(block)

    block = block_input
    filts = 32
    for j in range(2):
        block = UpSampling1D(4)(block)
        node_index = 1 - j
        block = concatenate([nodes[node_index], block])
        block_input = block
        block_output = []
        for i in range(3):
            block = Conv1D(filters=filts, kernel_size=9, padding='same')(block_input)
            block = Activation('elu')(block)
            block = BatchNormalization()(block)
            block_output.append(block)
            block_input = concatenate([block_input, block])

            block2 = Conv1D(filters=filts, kernel_size=9, padding='same', dilation_rate=4)(block_input)
            block2 = Activation('elu')(block2)
            block2 = BatchNormalization()(block2)
            block_output.append(block2)


            block3 = Conv1D(filters=filts, kernel_size=9, padding='same', dilation_rate=16)(block_input)
            block3 = Activation('elu')(block3)
            block3 = BatchNormalization()(block3)
            block_output.append(block3)

            block_input = concatenate([block_input, block, block2, block3])


        block = concatenate(block_output)
        filts //= 2



    outputs = Conv1D(filters=2, kernel_size=9, padding='same')(block_input)
    outputs = Activation('sigmoid')(outputs)
    outputs = Subtract()( [in_layer, outputs] )


    AutoEncoder = Model(in_layer, outputs)
    AutoEncoder.summary()
    return AutoEncoder

def discriminator(input_shape):
    in_layer = Input(input_shape)
    block_input = GaussianNoise(0.0)(in_layer)
    noisy_input = block_input
    filts = 2
    block = block_input
    for j in range(6):
        block_input = block
        block_output = []
        for i in range(1):
            block = Conv1D(filters=filts, kernel_size=9, padding='same')(block_input)
            block = Activation('elu')(block)
            block = BatchNormalization()(block)
            block_output.append(block)

            block2 = Conv1D(filters=filts, kernel_size=9, padding='same', dilation_rate=4)(block_input)
            block2 = Activation('elu')(block2)
            block2 = BatchNormalization()(block2)
            if j <= 4:
                block_output.append(block2)


            block3 = Conv1D(filters=filts, kernel_size=9, padding='same', dilation_rate=16)(block_input)
            block3 = Activation('elu')(block3)
            block3 = BatchNormalization()(block3)
            if j <= 2:
                block_output.append(block3)

            block_input = concatenate([block_input, block, block2, block3])

        if len(block_output) > 1:
            block = concatenate(block_output)
        else:
            block = block_output[0]
        block = Conv1D(filters=filts, kernel_size=9, padding='same', strides=4)(block)
        block = Activation('elu')(block)
        filts *= 2
        #block_input = BatchNormalization()(block)

    while filts > 1:
        for i in range(2):
            block = Conv1D(filters=filts, kernel_size=9, padding='same')(block_input)
            block = Activation('elu')(block)
            block_input = BatchNormalization()(block)
        filts = filts // 2

    block = Flatten()(block_input)
    units = 32
    while units > 2:
        block = Dense(units=units)(block)
        block = Activation('elu')(block)
        block = BatchNormalization()(block)
        units //= 2
    output_unit = Dense(units=1)(block)
    output_unit = Activation('sigmoid')(output_unit)
    discriminator = Model(in_layer, output_unit)
    #discriminator.summary()
    return discriminator

def gan_regularized_autoencoder(input_shape):
    in_layer = Input(input_shape)
    denoised_output = AutoEncoder(input_shape)(in_layer)
    discrim = discriminator(input_shape)
    #denoised_grad_flipped = gf.grad_flipper()(denoised_output)
    #discriminant_of_denoised = discrim(denoised_grad_flipped)
    discriminant_of_denoised = discrim(denoised_output)
    
    gan_regularized_autoencoder = Model(in_layer, [denoised_output, discriminant_of_denoised])
    gan_regularized_autoencoder.summary()
    plot_model(gan_regularized_autoencoder)
    return (gan_regularized_autoencoder, denoised_output, discrim)
## End AutoEncoder Model

if __name__ == '__main__':
    a = gan_regularized_autoencoder((20480, 2))
