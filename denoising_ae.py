import keras
import numpy as np
import tensorflow as tf
import os
import scipy.io.wavfile as wavio
import soundfile as sf

from keras.layers import Conv1D, Dense, Activation, BatchNormalization, GaussianNoise, Input
from keras.layers import concatenate, UpSampling1D
from keras.models import Model
from keras.optimizers import Nadam

def read_from_library(path, num_files='all'):
    files = os.listdir(path)
    song_arrs = []
    file_list = []
    for f in files:
        if 'flac' in f:
            file_list.append(f)
    if num_files == 'all':
        pass
    elif num_files >= len(file_list):
        pass
    else:
        file_list = file_list[0:num_files]

    print('loading {}'.format(file_list))

    for f in file_list:
        name = path + f
        data, sr = sf.read(name)
        sr = 20480 # slaps number, you can fit so many two's in here
        num_samples = data.shape[0] // sr
        song_arr = np.zeros((num_samples, sr, 2))
        for i in range(num_samples):
            start = i * sr
            end = start + sr
            song_arr[i, :, 0] = data[start:end, 0]
            song_arr[i, :, 1] = data[start:end, 1]
        song_arrs.append(song_arr)

    out_arr = np.concatenate(song_arrs, axis=0)
    return out_arr

def augment_snippets(snip_arr):
    num_samples = snip_arr.shape[0]
    out_snippets = np.zeros(snip_arr.shape)
    for i in range(num_samples):
        snippet = np.copy(snip_arr[i, :, :])
        snippet -= np.amin(snippet)
        snippet = snippet + np.random.normal(0, 0.1, snippet.shape)
        snippet = np.clip(snippet, 0.01, .99) 
        snippet -= 0.01
        snippet /= .98
        snippet -= np.mean(snippet)

        out_snippets[i, :, :] = snippet
    return out_snippets

## AutoEncoder Model
in_layer = Input((20480, 2))
block_input = GaussianNoise(0.0)(in_layer)
noisy_input = block_input
filts = 4
block = block_input
for j in range(2):
    block_input = block
    for i in range(4):
        block = Conv1D(filters=filts, kernel_size=9, padding='same')(block_input)
        block = Activation('elu')(block)
        block = BatchNormalization()(block)
        block_input = concatenate([block_input, block])

    filts *= 2
    block = Conv1D(filters=filts, kernel_size=9, padding='same', strides=2)(block)
    block = Activation('elu')(block)
    block_input = BatchNormalization()(block)

while filts > 4:
    for i in range(2):
        block = Conv1D(filters=filts, kernel_size=9, padding='same')(block_input)
        block = Activation('elu')(block)
        block = BatchNormalization()(block)
    filts = filts // 2

filts = 128
for j in range(2):
    block_input = block
    for i in range(4):
        block = Conv1D(filters=filts, kernel_size=9, padding='same')(block_input)
        block = Activation('elu')(block)
        block = BatchNormalization()(block)
        block_input = concatenate([block_input, block])

    filts //= 2
    block = UpSampling1D()(block)
    block = Activation('elu')(block)
    block_input = BatchNormalization()(block)

block_input = block
while filts > 1:
    filts = filts // 2
    for i in range(2):
        block = Conv1D(filters=filts, kernel_size=9, padding='same')(block_input)
        block = Activation('elu')(block)
        block = BatchNormalization()(block)
        block_input = concatenate([block_input, block])
outputs = Conv1D(filters=2, kernel_size=9, padding='same')(block_input)
outputs = Activation('sigmoid')(outputs)


autoencoder = Model(in_layer, noisy_input)
autoencoder.summary()
## End AutoEncoder Model

train_library_dir = '/home/john/Music/Floater/The Thief/'
val_library_dir = '/home/john/Music/Floater/Sink/'

train_snippets = read_from_library(train_library_dir, num_files=6).astype('float32')
val_snippets = read_from_library(val_library_dir, num_files=1).astype('float32')
norm_train_snippets = np.zeros((train_snippets.shape))
for i in range(norm_train_snippets.shape[0]):
    snip = np.copy(train_snippets[i, :, :])
    snip -= np.amin(snip)
    snip /= np.amax(snip)
    snip -= np.mean(snip)
    print('sum: {}, max: {}, min: {}'.format( np.sum(snip), np.amax(snip), np.amin(snip) ) )
    norm_train_snippets[i, :, :] = snip

norm_val_snippets = np.zeros((val_snippets.shape))
for i in range(val_snippets.shape[0]):
    snip = val_snippets[i, :, :]
    snip -= np.amin(snip)
    snip /= np.amax(snip)
    snip -= np.mean(snip)
    norm_val_snippets[i, :, :] = snip

autoencoder.compile(optimizer=Nadam(), loss='mean_absolute_error', metrics=['mean_squared_error'])

for i in range(1):
    aug_train_snippets = augment_snippets(norm_train_snippets)
    aug_val_snippets = augment_snippets(norm_val_snippets)

    autoencoder.fit(x=aug_train_snippets, y=norm_train_snippets, epochs=1, validation_data=(aug_val_snippets, norm_val_snippets), batch_size=4)







