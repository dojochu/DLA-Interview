from keras.layers import Input, Dense, Convolution2D, AveragePooling2D, MaxPooling2D, Flatten
from keras.models import Model
from audio_processor import create_batch_from_id
import pandas as pd
import numpy as np

output_path = '/home/ubuntu/dla/DLA/saved_model/'
data_ids = ['f','e','d','c','9','8','7','6','5','4','3','2','1','0']

'''
CONSTANTS
SR = Sample Rate
N_FFT = Parameter for Short Fourier Transforms
N_MELS = Number of Mel-Bins
HOP_LEN = Hop Length
DURA = Duration (to make it 1366 frame)
FRAME_SIZE = audio size
NUM_LABELS = Number of categories
TEST_SIZE = Test data_set size
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
BATCH_SIZE = 5

'''
Input Tensor - Using the Mel-Spectrogram of the .wav audio file to create a 96 x 1366 feature matrix.
input shape: (samples, channels, mel-bins, framesize) = (1, 1, 96, 1366).
'''
model_input = Input(shape=(1, N_MELS, FRAME_SIZE), name='initial_input', dtype='float32')

'''
Model Architecture
. 2D Convolutional Layer 64 feature maps of size 3x3
    input shape: (samples, channels, mel-bins, framesize) = (1, 1, 96, 1366)
    output shape: (samples, number of filters, rows, cols) = (1, 64, 96, 1366)
. 2D Convolutional Layer 128 feature maps of size 3x3
    input shape: (samples, number of filters, rows, cols) = (1, 64, 96, 1366)
    output shape: (samples, number of filters, rows, cols) = (1, 128, 96, 1366)
. 2D Average Pooling Layer pool size 2x2
    input shape: (samples, number of filters, rows, cols) = (1, 128, 96, 1366)
    output shape: (samples, number of filters, rows, cols) = (1, 128, 48, 683)
. 2D Convolutional Layer 96 feature maps of size 3x3
    input shape: (samples, number of filters, rows, cols) = (1, 128, 48, 683)
    output shape: (samples, number of filters, rows, cols) = (1, 96, 48, 683)
. 2D Average Pooling Layer pool size 4x4
    input shape: (samples, number of filters, rows, cols) = (1, 96, 48, 683)
    output shape: (samples, number of filters, rows, cols) = (1, 96, 12, 170)
. 2D Convolutional Layer 4 feature maps of size 3x3
    input shape: (samples, number of filters, rows, cols) = (1, 96, 12, 170)
    output shape: (samples, number of filters, rows, cols) = (1, 4, 12, 170)
. 2D Max Pooling Layer Pool size 4x17
    input shape: (samples, number of filters, rows, cols) = (1, 4, 12, 170)
    output shape: (samples, number of filters, rows, cols) = (1, 4, 3, 10)
. Flatten
    input shape: (samples, number of filters, rows, cols) = (1, 12, 3, 21)
    outputshape: (120, )
. Fully-Connected Layer
    input shape: output shape: (samples, number of filters, rows, cols) = (120,)
    output shape: (number of genres, ) = (188,)
'''

layer1 = Convolution2D(nb_filter=64, nb_row=3, nb_col=3,
                      init='glorot_normal', activation='relu', border_mode='same')
layer2 = Convolution2D(nb_filter=128, nb_row=3, nb_col=3,
                       init='glorot_normal', activation='relu', border_mode='same')
layer3 = MaxPooling2D(pool_size=(2, 2), border_mode='valid')
layer4 = Convolution2D(nb_filter=96, nb_row=3, nb_col=3,
                       init='glorot_normal', activation='relu',)
layer5 = MaxPooling2D(pool_size=(4, 4), border_mode='valid')
layer6 = Convolution2D(nb_filter=4, nb_row=3, nb_col=3,
                       init='glorot_normal', activation='relu',)
layer7 = AveragePooling2D(pool_size=(4, 17))
layer8 = Flatten()
layer9 = Dense(output_dim=NUM_LABELS, activation='relu')

x = layer1(model_input)
print(layer1.input_shape)
print(layer1.output_shape)
x = layer2(x)
print(layer2.output_shape)
x = layer3(x)
print(layer3.output_shape)
x = layer4(x)
print(layer4.output_shape)
x = layer5(x)
print(layer5.output_shape)
x = layer6(x)
print(layer6.output_shape)
x = layer7(x)
print(layer7.output_shape)
x = layer8(x)
print(layer8.output_shape)
model_output = layer9(x)
print(layer9.output_shape)


model = Model(input=model_input, output=model_output)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

total_songs = 0
total_batch = 0

for bid in data_ids[:1]:
    print('Training on data_id - ' + bid + ' - begins')
    melgrams, labels = create_batch_from_id(bid, 50, N_MELS,FRAME_SIZE, NUM_LABELS)

    SAMPLE_SIZE = melgrams.shape[0]

    for batch in [np.arange(0, SAMPLE_SIZE, 1)[a:b] for a, b in zip(np.arange(0, SAMPLE_SIZE, BATCH_SIZE), np.arange(BATCH_SIZE, SAMPLE_SIZE + BATCH_SIZE, BATCH_SIZE))]:
        total_songs += len(batch)
        total_batch += 1
        model.train_on_batch(melgrams[batch], labels[batch])
        print('Training on data_id - ' + bid + ' - completed...')
        print('')
model.save(output_path + 'CNN.h5')
print('Total Songs Trained: ' + str(total_songs))
print('Total Batches Trained: '+ str(total_batch))
#loss_and_metrics = model.evaluate(melgrams, labels)
#print(loss_and_metrics)
