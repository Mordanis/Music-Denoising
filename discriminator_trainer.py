import keras
import numpy as np
import tensorflow as tf
import os
import scipy.io.wavfile as wavio
import soundfile as sf
import gan_constructor
import sys
import grad_flipper as gf

from keras.utils import vis_utils
from keras import callbacks
from keras.layers import Conv1D, Dense, Activation, BatchNormalization, GaussianNoise, Input
from keras.layers import concatenate, UpSampling1D
from keras.models import Model
from keras.optimizers import Nadam, SGD

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

def get_batches(path_list, batch_size):
    trues = np.ones((batch_size, 1))
    false = np.zeros(trues.shape)
    path_pos = 0
    while True:
        path = path_list[path_pos]
        files = os.listdir(path)
        file_list = []
        for f in files:
            if 'flac' in f:
                file_list.append(f)
        for name in file_list:
            fname = os.path.join(path, name)
            song, sr = sf.read(fname)
            snips_per_sample = 20480
            num_snippets = song.shape[0] // snips_per_sample
            num_batches = num_snippets // batch_size
            song_arr = np.zeros((num_snippets, snips_per_sample, 2), dtype='float32')
            song_arr -= np.mean(song_arr)
            song_arr /= np.amax(song_arr).astype('float32')
            for ind in range(num_snippets):
                start = ind * snips_per_sample
                end = start + snips_per_sample
                song_arr[ind, :, :] = song[start:end, :]
            for b in range(num_batches):
                start = b * batch_size
                end = start + batch_size
                inputs = song_arr[start:end, :, :]
                noisy_inputs = augment_snippets(np.copy(inputs))
                inputs = np.concatenate([inputs, noisy_inputs], axis=0)
                outputs = np.concatenate([trues, false], axis=0)
                batch = (inputs, outputs)
                yield batch
        path_pos += 1
        if path_pos == len(path_list):
            path_pos = 0

def augment_snippets(snip_arr, noise_lvl='rand'):
    outputs = np.zeros(snip_arr.shape, dtype='float32')
    num_samples = snip_arr.shape[0]
    for i in range(num_samples):
        snippet = np.copy(snip_arr[i, :, :])
        if str(noise_lvl) == 'rand':
            noise_std = np.abs( np.random.rand() * 0.05)
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

def test_and_write(path, model, noise_level=None):
    clean_snips = read_from_library(path)
    snips = augment_snippets(clean_snips, noise_level)
    denoised_snips = model.predict([snips, snips])[0]
    denoised_pcm = snippets_to_pcm(denoised_snips)
    noisy_pcm = snippets_to_pcm(snips)

    sf.write('/home/john/Music/denoising-project/denoised/ganner.flac', denoised_pcm, samplerate=44100, format='flac', subtype='PCM_16')
    sf.write('/home/john/Music/denoising-project/noisy/ganner.flac', noisy_pcm, samplerate=44100, format='flac', subtype='PCM_16')
    return None



val_library_dirs = ['/home/extra_space/music/music/Music/Miles Davis/Kind Of Blue/']
train_library_dirs = ['/home/extra_space/music/music/sosobra/', '/home/extra_space/music/', 
                    '/home/extra_space/music/Snakes & Arrows/', '/home/extra_space/music/music/', '/home/extra_space/music/music/Music/wake/']

#autoencoder, generator, discriminator = gan_constructor.gan_regularized_autoencoder((20480, 2))
discrim_model = gan_constructor.discriminator((20480, 2))
discrim_model.summary()
#clean_in_layer = Input((20480, 2))
#discrim_of_real = discrim_model(clean_in_layer)
#noisy_in_layer = Input((20480, 2))
#discrim_of_noisy = discrim_model(noisy_in_layer)
#combined_model = Model([clean_in_layer, noisy_in_layer], [discrim_of_real, discrim_of_noisy])
#combined_model.summary()
#autoencoder = keras.models.load_model('denoiser.h5', custom_objects={'grad_flipper':gf.grad_flipper})
discrim_model.compile(optimizer=SGD(0.01), loss='binary_crossentropy', metrics=['mean_absolute_error'])
#vis_utils.plot_model(autoencoder, 'model.png', expand_nested=True)

train_generator = get_batches(train_library_dirs, 4)
val_generator = get_batches(val_library_dirs, 4)

def validation_generator():
    while True:
        next_batch = next(val_generator)
        out_batch = (np.concatenate(next_batch[0], axis=0), np.concatenate(next_batch[1], axis=0) )
        yield out_batch

ckpt = callbacks.ModelCheckpoint('discrim.h5', verbose=1, save_best_only=True)
for i in range(100):
    #discrim_model.fit_generator(train_generator, steps_per_epoch=1500, epochs=1 + i, initial_epoch=i,
    #                           callbacks=[ckpt], validation_data=val_generator, validation_steps=500)

    epoch_str = 'epoch {}: '.format(i + 1)
    sys.stdout.write(epoch_str)
    sys.stdout.flush()
    num_steps = 0
    metrics_dict = {}
    for metric in discrim_model.metrics_names:
        metrics_dict[metric] = []
    for batch in train_generator:
        progress_ratio = num_steps / 1500
        num_dashes = int(np.round(100 * progress_ratio)) - 1
        num_spaces = int(np.round(100 * (1 - progress_ratio)))
        dashes = num_dashes * '-'
        spaces = num_spaces * ' '
        progress_bar = '\r' + epoch_str + '[' + dashes + '>' + spaces + '] {}'.format(num_steps)
        num_steps += 1
        inputs = batch[0]
        outputs = batch[1]
        losses = discrim_model.train_on_batch(inputs, outputs)

        for ind in range(len(discrim_model.metrics_names)):
            metrics_dict[discrim_model.metrics_names[ind]].append(losses[ind])
        metrics_string = ''
        for key in metrics_dict.keys():
            metrics_string += key + ': {}\t'.format(np.mean(metrics_dict[key]))

        sys.stdout.write(progress_bar + metrics_string)
        sys.stdout.flush()

        if num_steps == 1500:
            break

    sys.stdout.write('\n')
    sys.stdout.flush()
    losses = discrim_model.evaluate_generator(val_generator, steps=100, verbose=1)
    metrics_string = ''
    for i in range(len(discrim_model.metrics_names)):
        name = discrim_model.metrics_names[i]
        metric = np.mean(losses[i])
        metrics_string += (name + ': {}\t'.format(metric))
    print(metrics_string)


    #test_and_write(val_library_dirs[0], autoencoder)


