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

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

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

def get_batches(path_list, batch_size, generator):
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
                denoised_inputs = generator.predict(noisy_inputs)[0]
                batch = ([noisy_inputs, inputs, denoised_inputs], [inputs, false, false, trues, false])
                yield batch
        path_pos += 1
        if path_pos == len(path_list):
            path_pos = 0

def split_batches(path_list, batch_size):
    gen = get_batches(path_list, batch_size, autoencoder)
    while True:
        batch = next(gen)
        autoencoder_inputs = batch[0][0]
        autoencoder_outputs = [batch[1][0], batch[1][3]]
        discrim_inputs = np.concatenate(batch[0], axis=0)
        #discrim_inputs = batch[0][1]
        discrim_outputs = np.concatenate(batch[1][2:], axis=0)
        #discrim_outputs = batch[1][3]
        yield ( (autoencoder_inputs, autoencoder_outputs), (discrim_inputs, discrim_outputs) )

def get_autoencoder_batches(path_list, batch_size):
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
                noisy_inputs = augment_snippets(np.copy(inputs), noise_lvl=0.1)
                batch = (noisy_inputs, inputs)
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
            noise_std = np.abs( np.random.rand() * 0.1)
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

def snippets_to_pcm(snippets):
    snips = snippets.copy()
    snips -= np.amin(snips)
    snips /= np.amax(snips)
    snips -= 0.5
    snips *= (32000 * 2)
    snips -= np.mean(snips)

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
    denoised_snips = model.predict(snips, batch_size=3)[0]
    denoised_pcm = snippets_to_pcm(denoised_snips)
    noisy_pcm = snippets_to_pcm(snips)

    sf.write('/home/tim/Music/denoising-project/denoised/autoencoder.flac', denoised_pcm, samplerate=44100, format='flac', subtype='PCM_16')
    sf.write('/home/tim/Music/denoising-project/noisy/autoencoder.flac', noisy_pcm, samplerate=44100, format='flac', subtype='PCM_16')
    return None


test_dir = '/home/tim/Music/Robben Ford/'
val_library_dirs = ['/home/tim/Music/Three Days Grace/One-X/']
train_library_dirs = ['/home/tim/Music/Rush/', '/home/tim/Music/Floater/Sink/', '/home/tim/Music/Floater/The Thief/']

autoencoder, generator, discriminator = gan_constructor.gan_regularized_autoencoder((20480, 2))
#autoencoder.layers[0] = keras.models.load_model('denoiser.h5', custom_objects={'grad_flipper':gf.grad_flipper})

discriminator.trainable = True
discriminator.compile(optimizer=Nadam(), loss='binary_crossentropy', metrics=['mean_absolute_error'])
autoencoder.layers[1].trainable = True 
autoencoder.layers[-1].trainable = False
autoencoder.compile(optimizer=Nadam(), loss={'model_1':'mean_absolute_error', 'model_2':'binary_crossentropy'},
                    metrics=['mean_squared_error'], loss_weights=[1, 0.005])

print('discriminator is trainable? ' + str(discriminator.trainable))
print('discriminator in combined model trainable? ' + str(autoencoder.layers[-1].trainable))
#autoencoder = gan_constructor.AutoEncoder((20480, 2))
#vis_utils.plot_model(autoencoder, 'model.png', expand_nested=True)

train_generator = split_batches(train_library_dirs, 3)
val_generator = split_batches(val_library_dirs, 3)

ckpt = callbacks.ModelCheckpoint('denoiser.h5', verbose=1, save_best_only=True)
tboard = callbacks.TensorBoard()
#autoencoder.compile(optimizer=Nadam(), loss='mean_absolute_error', metrics=['mean_squared_error'])
for i in range(100):

    epoch_str = 'epoch {}: '.format(i + 1)
    sys.stdout.write(epoch_str)
    sys.stdout.flush()
    num_steps = 0
    discrim_metrics_dict = {}
    gen_metrics_dict = {}
    for metric in discriminator.metrics_names:
        discrim_metrics_dict[metric] = []
    for metric in autoencoder.metrics_names:
        gen_metrics_dict[metric] = []
    for batch in train_generator:
        autoencoder_batch = batch[0]
        discriminator_batch = batch[1]

        progress_ratio = num_steps / 1500
        num_dashes = int(np.round(25 * progress_ratio)) - 1
        num_spaces = int(np.round(25 * (1 - progress_ratio)))
        dashes = num_dashes * '-'
        spaces = num_spaces * ' '
        progress_bar = '\r' + epoch_str + '[' + dashes + '>' + spaces + '] {} '.format(num_steps + 1)
        num_steps += 1

        for _ in range(1):
            losses = discriminator.train_on_batch(discriminator_batch[0], discriminator_batch[1])
        for ind in range(len(discriminator.metrics_names)):
            discrim_metrics_dict[discriminator.metrics_names[ind]].append(losses[ind])
        metrics_string = ''
        for key in discrim_metrics_dict.keys():
            metrics_string += 'dis-' + key + ':{:.3e} '.format(np.mean(discrim_metrics_dict[key]))

        #autoencoder.layers[-1].trainable = False
        losses = autoencoder.train_on_batch(autoencoder_batch[0], autoencoder_batch[1])
        #autoencoder.layers[-1].trainable = True

        for ind in range(len(autoencoder.metrics_names)):
            gen_metrics_dict[autoencoder.metrics_names[ind]].append(losses[ind])
        for key in gen_metrics_dict.keys():
            metrics_string += 'gen-' + key + ':{:.3e} '.format(np.mean(gen_metrics_dict[key]))

        sys.stdout.write(progress_bar + metrics_string)
        sys.stdout.flush()

        if num_steps == 1500:
            break

    sys.stdout.write('\n')
    sys.stdout.flush()

    discrim_metrics_dict = {}
    gen_metrics_dict = {}
    for metric in discriminator.metrics_names:
        discrim_metrics_dict[metric] = []
    for metric in autoencoder.metrics_names:
        gen_metrics_dict[metric] = []
    for i in range(100):
        batch = next(val_generator)
        autoencoder_batch = batch[0]
        discriminator_batch = batch[1]
        losses = discriminator.test_on_batch(discriminator_batch[0], discriminator_batch[1])
        for ind in range(len(discriminator.metrics_names)):
            discrim_metrics_dict[discriminator.metrics_names[ind]].append(losses[ind])
        metrics_string = '\rvalidation: '
        for key in discrim_metrics_dict.keys():
            metrics_string += 'dis-' + key + ':{:.3e} '.format(np.mean(discrim_metrics_dict[key]))

        losses = autoencoder.train_on_batch(autoencoder_batch[0], autoencoder_batch[1])
        for ind in range(len(autoencoder.metrics_names)):
            gen_metrics_dict[autoencoder.metrics_names[ind]].append(losses[ind])
        for key in gen_metrics_dict.keys():
            metrics_string += 'gen-' + key + ':{:.3e} '.format(np.mean(gen_metrics_dict[key]))
        sys.stdout.write(metrics_string)
        sys.stdout.flush()
    print(metrics_string[1:])
    test_and_write(test_dir, autoencoder)
    autoencoder.save('autoencoder.h5')
    discriminator.save('discriminator.h5')
fake_in_layer = Input((20480, 2))
fake_model = Model(fake_in_layer, fake_in_layer)
fake_model.compile(optimizer=Nadam(), loss='mean_squared_error', metrics=['mean_absolute_error'])
fake_model.fit_generator(train_generator, steps_per_epoch=1500, epochs=100, initial_epoch=0,
                         validation_data=val_generator, validation_steps=500)

'''

exit(1)i




for epoch in range(100):
    autoencoder.fit_generator(train_generator, steps_per_epoch=1500, epochs=1+epoch, initial_epoch=epoch,
                          callbacks=[ckpt, tboard], validation_data=val_generator, validation_steps=500)
    test_and_write(test_dir, autoencoder)

'''

