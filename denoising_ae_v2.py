import keras
import numpy as np
import tensorflow as tf
import os
import scipy.io.wavfile as wavio
import soundfile as sf
import gc

from keras.layers import Conv1D, Dense, Activation, BatchNormalization, GaussianNoise, Input
from keras.layers import concatenate, UpSampling1D, Subtract
from keras.models import Model
from keras.optimizers import Nadam
from keras import callbacks

def read_from_library(path, num_files='all'):
    files = os.listdir(path)
    song_arrs = []
    file_list = []
    for f in files:
        if len(f) > 5 and f[-4:] == 'flac':
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

def augment_snippets(snip_arr, noise_lvl='rand'):
    outputs = np.zeros(snip_arr.shape, dtype='float32')
    num_samples = snip_arr.shape[0]
    for i in range(num_samples):
        snippet = np.copy(snip_arr[i, :, :])
        if str(noise_lvl) == 'rand':
            noise_std = np.abs( np.random.rand() * 0.01 )
        elif noise_lvl is None:
            noise_std = 0
        else:
            noise_std = noise_lvl
        snippet = snippet + np.random.normal(0, noise_std, snippet.shape)
        #snippet = np.clip(snippet, 0.01, .99) 
        #snippet -= 0.01
        #snippet /= .98
        outputs[i, :, :] = snippet
    return outputs

def snippets_to_pcm(snips):
    snips = np.copy(snips)
    snips -= np.amin(snips)
    snips /= np.amax(snips)
    snips -= 0.5
    snips *= (32767 * 2)
    snips = snips.astype('int16')
    num_snippets = snips.shape[0]
    samples_per_snip = snips.shape[1]
    out_shape = (num_snippets * samples_per_snip, 2)
    pcm = np.zeros(out_shape, dtype='int16')
    for i in range(num_snippets):
        start = samples_per_snip * i
        end = start + samples_per_snip
        pcm[start:end, :] = snips[i, :, :]
    return pcm


## AutoEncoder Model
in_layer = Input((20480, 2))
block_input = GaussianNoise(0.0)(in_layer)
noisy_input = block_input
filts = 8
block = block_input
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
    block = Conv1D(filters=filts * 3, kernel_size=9, padding='same', strides=2)(block)
    block = Activation('elu')(block)
    filts *= 2
    block_input = BatchNormalization()(block)

while filts > 4:
    for i in range(2):
        block = Conv1D(filters=filts, kernel_size=9, padding='same')(block_input)
        block = Activation('elu')(block)
        block = BatchNormalization()(block)
    filts = filts // 2

filts = 32
for j in range(2):
    block_input = block
    block_output = []
    for i in range(2):
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
    block = UpSampling1D()(block)

block_input = block

for i in range(2):
    block = Conv1D(filters=filts, kernel_size=9, padding='same')(block_input)
    block = Activation('elu')(block)
    block = BatchNormalization()(block)
    block_output.append(block)
    block_input = concatenate([block_input, block])

outputs = Conv1D(filters=2, kernel_size=9, padding='same')(block_input)
outputs = Activation('tanh')(outputs)
outputs = Subtract()( [in_layer, outputs] )


autoencoder = Model(in_layer, outputs)
autoencoder.summary()
## End AutoEncoder Model

val_library_dir = '/home/extra_space/music/music/Music/Miles Davis/Kind Of Blue/'
train_library_dirs = ['/home/extra_space/music/music/sosobra/', '/home/extra_space/music/', 
                    '/home/extra_space/music/Snakes & Arrows/', '/home/extra_space/music/music/', '/home/extra_space/music/music/Music/wake/']
train_library_pos = 0
train_library_dir = train_library_dirs[train_library_pos]

train_snippets = read_from_library(train_library_dir).astype('float32')
val_snippets = read_from_library(val_library_dir, num_files=1).astype('float32')
outputs = snippets_to_pcm(val_snippets)
sf.write(file='/home/john/Music/denoising-project/baseline/combined.flac', data=snippets_to_pcm(val_snippets), samplerate=44100, format='flac', subtype='PCM_16')

val_snippets -= np.amin(val_snippets)
val_snippets /= np.amax(val_snippets)
val_snippets -= 0.5
train_snippets -= np.amin(train_snippets)
train_snippets /= np.amax(train_snippets)
train_snippets -= 0.5

autoencoder = keras.models.load_model('denoiser.h5')
autoencoder.compile(optimizer=Nadam(), loss='mean_absolute_error', metrics=['mean_squared_error'])

ckpt = callbacks.ModelCheckpoint('denoiser.h5', verbose=1, save_best_only=True)
for i in range(100):
    aug_train_snippets = augment_snippets(train_snippets)
    aug_val_snippets = augment_snippets(val_snippets, noise_lvl=None)
    autoencoder.fit(x=aug_train_snippets, y=train_snippets, epochs=i+1, validation_data=(aug_val_snippets, val_snippets), batch_size=4, initial_epoch=i, callbacks=[ckpt])


    denoised_val_snippets = autoencoder.predict(aug_val_snippets)
    denoised_outputs = snippets_to_pcm(denoised_val_snippets)
    sf.write('/home/john/Music/denoising-project/denoised/combined.flac', denoised_outputs, samplerate=44100, format='flac', subtype='PCM_16')
    noisy_outputs = snippets_to_pcm(aug_val_snippets)
    sf.write('/home/john/Music/denoising-project/noisy/combined.flac', noisy_outputs, samplerate=44100, format='flac', subtype='PCM_16')
    clean_outputs = snippets_to_pcm(val_snippets)
    sf.write('/home/john/Music/denoising-project/target/combined.flac', clean_outputs, samplerate=44100, format='flac', subtype='PCM_16')

    train_library_pos += 1
    train_snippets = 'a'
    train_aug_snippets = 'b'
    gc.collect()

    if train_library_pos == len(train_library_dirs):
        train_library_pos = 0
    train_library_dir = train_library_dirs[train_library_pos]
    train_snippets = read_from_library(train_library_dir).astype('float32')
    train_snippets -= np.amin(train_snippets)
    train_snippets /= np.amax(train_snippets)
    train_snippets -= 0.5

