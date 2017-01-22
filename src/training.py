from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, Flatten
from keras.models import Model
from audio_processor import create_batch_from_id, get_clips_from_id
import numpy as np
import sys

output_path = '/home/ubuntu/dla/DLA/saved_model/'
data_ids = ['9', '8', '7', '6', '4', '3', '2', '1', '0']
clip_train = get_clips_from_id(data_ids)


'''
CONSTANTS
SR = Sample Rate
N_FFT = Length of segment for Short-Time Fourier Transform
N_MELS = Frequency bin scale
HOP_LEN = Hop length
DURA = Duration
FRAME_SIZE = Data vector size
NUM_LABELS = Number of categories
TEST_SIZE = Test dataset size
BATCH_SIZE = batch size
'''

SR = 12000
N_FFT = 512
N_MELS = 96
HOP_LEN = 256
DURA = 29.12
FRAME_SIZE = int(DURA * SR / HOP_LEN + 1)
NUM_LABELS = 188
TEST_SIZE = 2000
BATCH_SIZE = int(sys.argv[1])

'''
Input Tensor - Using the Mel-Spectrogram of the .wav audio file to create a N_MELS x FRAME_SIZE feature matrix.
input shape: (samples, channels, mel-bins, framesize) = (1, 1, N_MELS, FRAME_SIZE).
'''
print('Defining input vector...')
model_input = Input(shape=(1, N_MELS, FRAME_SIZE), name='initial_input', dtype='float32')


'''
Building the architecture of the Convolutional 2-dimensional neural network
'''

print('Defining Deep Learning Model...')
layer1 = Convolution2D(nb_filter=32, nb_row=3, nb_col=3,
                      init='glorot_normal', activation='relu', border_mode='same')
layer2 = Convolution2D(nb_filter=64, nb_row=3, nb_col=3,
                       init='glorot_normal', activation='relu', border_mode='same')
layer3 = MaxPooling2D(pool_size=(2, 2), border_mode='valid')
layer4 = Convolution2D(nb_filter=32, nb_row=3, nb_col=3,
                       init='glorot_normal', activation='relu', border_mode='same')
layer5 = MaxPooling2D(pool_size=(2, 2), border_mode='valid')
layer6 = Convolution2D(nb_filter=16, nb_row=3, nb_col=3,
                       init='glorot_normal', activation='relu', border_mode='same')
layer7 = MaxPooling2D(pool_size=(2, 2), border_mode='valid')
layer8 = Convolution2D(nb_filter=8, nb_row=3, nb_col=3,
                       init='glorot_normal', activation='relu', border_mode='same')
layer9 = MaxPooling2D(pool_size=(2, 2), border_mode='valid')
layer10 = Convolution2D(nb_filter=8, nb_row=3, nb_col=3,
                       init='glorot_normal', activation='relu', border_mode='same')
layer11 = MaxPooling2D(pool_size=(4, 4), border_mode='valid')
layer12 = Flatten()
layer13 = Dense(output_dim=NUM_LABELS, activation='softmax')

x = layer1(model_input)
x = layer2(x)
x = layer3(x)
x = layer4(x)
x = layer5(x)
x = layer6(x)
x = layer7(x)
x = layer8(x)
x = layer9(x)
x = layer10(x)
x = layer11(x)
x = layer12(x)
x = layer13(x)
model_output = x

print(layer1.input_shape)
print(layer1.output_shape)
print(layer2.output_shape)
print(layer3.output_shape)
print(layer4.output_shape)
print(layer5.output_shape)
print(layer7.output_shape)
print(layer6.output_shape)
print(layer7.output_shape)
print(layer8.output_shape)
print(layer9.output_shape)
print(layer10.output_shape)
print(layer11.output_shape)

print('Compiling model...')
model = Model(input=model_input, output=model_output)
model.compile(optimizer='Adadelta', loss='categorical_crossentropy', metrics=['categorical_crossentropy', 'cosine_proximity'])


'''
Run each training/gradient step on small batch of data

'''

total_songs = 0
total_batch = 0

SAMPLE_SIZE = clip_train.shape[0]
print('SAMPLE_SIZE: ' + str(SAMPLE_SIZE))
print('Beginning training...')

for batch in [np.arange(0, SAMPLE_SIZE, 1)[a:b] for a, b in zip(np.arange(0, SAMPLE_SIZE, BATCH_SIZE), np.arange(BATCH_SIZE, SAMPLE_SIZE + BATCH_SIZE, BATCH_SIZE))]:

    try:
        melgrams, labels = create_batch_from_id(clips=clip_train.iloc[batch], np_cache=True, N_MELS=N_MELS, SR=SR, N_FFT=N_FFT, HOP_LEN=HOP_LEN, DURA=DURA)
        total_batch += 1
        total_songs += melgrams.shape[0]
        print('Training on batch number: ' + str(total_batch))
        print('Training on ' + str(melgrams.shape[0]) + ' songs')
        print('Total songs: ' + str(total_songs))
        model.train_on_batch(melgrams, labels)
    except Exception as e:
        print('Error occured: batch ' + str(total_batch) + ' skipped')
        continue

print('')
model.save(output_path + 'CNN3.h5')
print('Total Songs Trained: ' + str(total_songs))
print('Total Batches Trained: ' + str(total_batch))

