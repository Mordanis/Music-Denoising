import keras
import numpy as np
import tensorflow as tf
import os
import scipy.io.wavfile as wavio
import soundfile as sf
import gan_constructor
import sys

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

def get_batches(path_list, batch_size):

    for path in path_list:
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
                batch = song_arr[start:end, :, :]
                yield batch


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

train_library_dirs = ['/home/john/Music/Floater/The Thief/']
val_library_dirs = ['/home/john/Music/Floater/Sink/']

autoencoder, generator, discriminator = gan_constructor.gan_regularized_autoencoder((20480, 2))

autoencoder.compile(optimizer=Nadam(), loss='mean_absolute_error', metrics=['mean_squared_error'])
discriminator.trainable = True
discriminator.compile(optimizer=Nadam(), loss='mean_absolute_error', metrics=['mean_squared_error'])
batch_size = 4
train_trues = np.ones((batch_size, 1))
train_false = np.zeros(train_trues.shape)
val_trues = np.ones((batch_size, 1))
val_false = np.zeros(val_trues.shape)



for epoch in range(15):
    gen_metrics = None
    true_metrics = None
    false_metrics = None
    sys.stdout.write('training epoch: {} '.format(epoch))
    for train_batch in get_batches(train_library_dirs, batch_size):
        bad_batch = augment_snippets(train_batch)
        metrics = autoencoder.train_on_batch(bad_batch, [train_batch, train_trues, train_trues])
        if gen_metrics is not None:
            metrics = np.expand_dims(metrics, axis=0)
            gen_metrics = np.append(gen_metrics, metrics, axis=0)
        else: 
            gen_metrics = np.expand_dims(np.array(metrics), axis=0)

        reconstructed = autoencoder.predict(bad_batch)[0]
        discriminator.trainable = True
        true_discrim_metrics = discriminator.train_on_batch(train_batch, train_trues)
        false_discrim_metrics = discriminator.train_on_batch(reconstructed, train_false)
        discriminator.trainable = False

        ''' *** UNUSED ***
        if true_metrics is not None:
            true_metrics = np.expand_dims(true_discrim_metrics, axis=0)
            true_metrics = np.append(true_metrics, true_discrim_metrics, axis=0)
        else:
            true_metrics = np.expand_dims(np.array(true_discrim_metrics), axis=0)
        if false_metrics is not None:
            false_metrics = np.expand_dims(false_discrim_metrics,axis=0)
            false_metrics.append(false_discrim_metrics, axis=0)
        else:
            false_metrics = np.expand_dims(np.array(false_discrim_metrics), axis=0)
        '''
        metrics = np.mean(gen_metrics, axis=0)
        metrics_string = '\r\repoch: {} '.format(epoch)
        for pos in range(len(autoencoder.metrics_names)):
            name = autoencoder.metrics_names[pos]
            metrics_string += 'gen-'
            metrics_string += name
            metrics_string += ': {} '.format(metrics[pos])
        '''
        t_metrics = np.mean(true_metrics, axis=0)
        f_metrics = np.mean(false_metrics, axis=0)
        for pos in range(len(discriminator.metrics_names)):
            name = discriminator.metrics_names[pos]
            metrics_string += 'discrim-true-'
            metrics_string += name
            metrics_string += ': {} '.format(t_metrics[pos])
            metrics_string += 'discrim-false-'
            metrics_string += name
            metrics_string += ': {} '.format(f_metrics[pos])
        '''
        sys.stdout.flush()
        sys.stdout.write(metrics_string)




    gen_metrics = None
    true_metrics = None
    false_metrics = None
    sys.stdout.write('validation epoch: {} '.format(epoch))
    for val_batch in get_batches(val_library_dirs, batch_size):
        bad_batch = augment_snippets(val_batch)
        metrics = autoencoder.test_on_batch(bad_batch, val_batch)

        if gen_metrics is not None:
            metrics = np.expand_dims(metrics, axis=0)
            gen_metrics = np.append(gen_metrics, metrics, axis=0)
        else: 
            gen_metrics = np.expand_dims(np.array(metrics), axis=0)


        '''
        reconstructed = autoencoder.predict(bad_batch)[0]
        true_discrim_metrics = autoencoder.validate_on_batch(train_batch, train_trues)
        false_discrim_metrics = autoencoder.validate_on_batch(reconstructed, train_false)
        if true_metrics is not None:
            true_metrics = np.expand_dims(true_discrim_metrics, axis=0)
            true_metrics = np.append(true_metrics, true_discrim_metrics, axis=0)
        else:
            true_metrics = np.expand_dims(np.array(false_discrim_metrics), axis=0)
        if false_metrics is not None:
            false_metrics = np.expand_dims(false_discrim_metrics, axis=0)
            false_metrics = np.append(false_metrics, false_discrim_metrics, axis=0)
        else:
            false_metrics = np.expand_dims(np.array(false_discrim_metrics), axis=0)
        '''
        metrics = np.mean(gen_metrics, axis=0)
        metrics_string = '\r\repoch: {}'.format(epoch)
        for pos in range(len(autoencoder.metrics_names)):
            name = autoencoder.metrics_names[pos]
            metrics_string += 'gen-'
            metrics_string += name
            metrics_string += ': {} '.format(metrics[pos])
        '''
        t_metrics = np.mean(true_metrics, axis=0)
        f_metrics = np.mean(false_metrics, axis=0)
        for pos in range(len(discriminator.metrics_names)):
            name = discriminator.metrics_names[pos]
            metrics_string += 'discrim-true-'
            metrics_string += name
            metrics_string += ': {} '.format(t_metrics[pos])
            metrics_string += 'discrim-false-'
            metrics_string += name
            metrics_string += ': {} '.format(f_metrics[pos])
        '''
        sys.stdout.flush()
        sys.stdout.write(metrics_string)






