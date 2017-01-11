from keras.layers import Input, Dense
from keras.models import Model
from audio_processor import compute_melgram
import pandas as pd
import numpy as np

data_path = '../music_samples/MoShang BEATS/'
label_path = '../music_samples/annotations_final.csv'
info_path = '../music_samples/clip_info_final.csv'

songs = ['90_B04.mp3', '90_M03.mp3', '90_M07.mp3', '90_M10.mp3']

clip_info = pd.read_csv(filepath_or_buffer=info_path, delimiter='\t')
label_info = pd.read_csv(filepath_or_buffer=label_path, delimiter='\t')
clip_info['batch_id'] = pd.Series([str(b)[2] for b in clip_info.mp3_path.str.split('/')])
label_info['batch_id'] = pd.Series([str(b)[2] for b in label_info.mp3_path.str.split('/')])

SR = 12000
N_FFT = 512
N_MELS = 96
HOP_LEN = 256
DURA = 29.12  # to make it 1366 frame..
FRAME_SIZE = int(N_MELS * (DURA * SR / HOP_LEN + 1))
NUM_LABELS = 188
TEST_SIZE = 2000
iterations = 1000


#model_input = tf.placeholder(dtype='float32', shape=(FRAME_SIZE, 1))
model_input = Input(shape=(FRAME_SIZE,))
#truth = tf.placeholder(dtype='float32', shape=(NUM_LABELS, 1))
m = Dense(50, activation='softmax')(model_input)
m = Dense(50, activation='softmax')(m)
model_output = Dense(NUM_LABELS, activation='softmax')(m)

model = Model(input=model_input, output=model_output)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


for bid in pd.unique(clip_info.batch_id):
    melgrams = np.zeros(shape=(FRAME_SIZE,))
    print(melgrams)
    labels = np.zeros(shape=(NUM_LABELS,))
    #for song in clip_info[clip_info['batch_id'] == bid]['mp3_path']:
    for song in songs:
        melgrams = np.vstack((melgrams, compute_melgram(data_path + song).flatten()))
        #melgrams = compute_melgram(data_path + song).transpose().flatten()
        #print(melgrams.shape)
        #print(model_input)
        #labels = np.concatenate(labels, label_info[label_info['mp3_path'] == song].filter(
        #                      items=label_info.columns.tolist()[1:-1]).transpose())
        labels = np.vstack((labels, np.zeros(shape=(NUM_LABELS,))))
        #labels = np.zeros(shape=(NUM_LABELS,))
        model.train_on_batch(melgrams, labels)

test_x = np.vstack((np.zeros(shape=(FRAME_SIZE,)), compute_melgram(data_path + songs[0]).flatten()))
test_y = np.vstack((np.zeros(shape=(NUM_LABELS,)),np.ones(shape=(NUM_LABELS,))))
loss_and_metrics = model.evaluate(test_x, test_y)


print(loss_and_metrics)
