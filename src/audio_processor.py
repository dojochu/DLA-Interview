# compute_melgram credit to https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/audio_processor.py

import librosa
import numpy as np
import pandas as pd

d_path = '/media/ubuntu/9C33-6BBD/'
l_file = '/media/ubuntu/9C33-6BBD/annotations_final.csv'
m_file = '/media/ubuntu/9C33-6BBD/clip_info_final.csv'

#Some default parameters
n_mels = 96
sr = 12000
n_fft = 512
hop_len = 256
dura = 29.12
n_labels = 188

#Reading in metadata and label data
clip_info = pd.read_csv(filepath_or_buffer=m_file, delimiter='\t')
clip_info['batch_id'] = pd.Series([str(b)[2] for b in clip_info.mp3_path.str.split('/')])
label_info = pd.read_csv(filepath_or_buffer=l_file, delimiter='\t')


def get_clips_from_id(data_ids):
    '''
    Function: get clip_ids based on batch ids

    Arguments

        data_id: ids from the batch data (f,e,d,c,0,1,2,3,4,5,6,7,8,9)

    Return: Pandas dataframe of clip_ids

    '''
    return clip_info[clip_info['batch_id'].isin(data_ids)]['clip_id']


def get_clips_from_label(label):

    '''
    Function: get clip_ids based on labels

    Arguments

        label: music class category

    Return Pandas dataframe of clip_ids
    '''

    return label_info[label_info[label] == 1]['clip_id']


def create_batch_from_id(clips, np_cache=True, N_MELS=n_mels, N_FFT=n_fft, HOP_LEN=hop_len, DURA=dura, SR=sr, NUM_LABELS=n_labels, data_path=d_path, label_file=l_file, metadata_file=m_file):

    '''
    Function: Get Mel-Spectrogram and label data

    Arguments

        clips: from get get_clip functions
        np_cache: Use mp3 data that was cached so you don't have to run the mel-spectrogram calcultions again

    Return: Numpy ndarray Mel-spectrogram, Numpy ndarray Label

    '''

    melgrams = np.zeros(shape=(1, 1, N_MELS, int(DURA * SR / HOP_LEN + 1),))
    labels = np.zeros(shape=(NUM_LABELS,))
    song_list = clip_info[clip_info['clip_id'].isin(clips)]['mp3_path']

    num_song = 0
    for song in song_list:
        try:
            num_song += 1
            if np_cache:
                melgrams = np.concatenate((melgrams, np.load(data_path + 'npy/' + song.split('.')[0] + '.npy')))
                labels = np.vstack((labels, np.load(data_path + 'npy/' + song.split('.')[0] + '_label.npy')))

            else:
                melgrams = np.concatenate((melgrams, compute_melgram(data_path + 'mp3/' + song, SR=SR, N_FFT=N_FFT, N_MELS=N_MELS, HOP_LEN=HOP_LEN, DURA=DURA)))
                labels = np.vstack((labels, label_info[label_info['mp3_path'] == song].filter(items=label_info.columns.tolist()[1:-1]).transpose().iloc[:, 0]))
        except:
            print('song ' + song + 'caused an error')
            continue
    return melgrams[1:], labels[1:]


def compute_melgram(audio_path, SR=sr, N_FFT=n_fft, N_MELS=n_mels, HOP_LEN=hop_len, DURA=dura):

    '''
    Author: Keun Woo Choi

    Function: compute Mel-spectrogram

    Arguments:
        audio_path: path of audio file

    Return: Numpy ndarray of Mel-spectrogram

    '''
    src, sr = librosa.load(audio_path, sr=SR)  # whole signal
    n_sample = src.shape[0]
    n_sample_fit = int(DURA * SR)

    if n_sample < n_sample_fit:  # if too short
        src = np.hstack((src, np.zeros((int(DURA * SR) - n_sample,))))
    elif n_sample > n_sample_fit:  # if too long
        src = src[int((n_sample - n_sample_fit) / 2):int((n_sample + n_sample_fit) / 2)]
    logam = librosa.logamplitude
    melgram = librosa.feature.melspectrogram
    ret = logam(melgram(y=src, sr=SR, hop_length=HOP_LEN, n_fft=N_FFT,
                n_mels=N_MELS) ** 2, ref_power=1.0)
    ret = ret[np.newaxis, np.newaxis, :]

    return ret
