# From keunwoochoi: https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/audio_processor.py

import librosa
import numpy as np
import pandas as pd
import sys

d_path = '/media/ubuntu/9C33-6BBD/mp3/'
l_file = '/media/ubuntu/9C33-6BBD/annotations_final.csv'
m_file = '/media/ubuntu/9C33-6BBD/clip_info_final.csv'
o_path = '/media/ubuntu/9C33-6BBD/npy/'

mel_bins = 96
frames = 1366
n_labels = 188



def create_batch_from_id(bid, first_num, N_MELS=mel_bins, FRAME_SIZE=frames, NUM_LABELS=n_labels, data_path=d_path, label_file=l_file, metadata_file=m_file):
   
    clip_info = pd.read_csv(filepath_or_buffer=metadata_file, delimiter='\t')
    label_info = pd.read_csv(filepath_or_buffer=label_file, delimiter= '\t')
    clip_info['batch_id'] = pd.Series([str(b)[2] for b in clip_info.mp3_path.str.split('/')])
    label_info['batch_id'] = pd.Series([str(b)[2] for b in label_info.mp3_path.str.split('/')])

    print('Creating data on batch_id = ' + bid + '- begins')
    song_list = clip_info[clip_info['batch_id'] == bid]['mp3_path']
    
    if first_num is None:
        first_num = song_list.shape[0]
    num_song = 0
    for song in song_list.iloc[:first_num]:
        try:
            num_song += 1
            melgram = compute_melgram(data_path + song)
            label = label_info[label_info['mp3_path'] == song].filter(items=label_info.columns.tolist()[1:-2]).transpose().iloc[:,0]
            np.save(o_path + song.split('.')[0], melgram)
            np.save(o_path + song.split('.')[0]+'_label', label)
        except FileNotFoundError:
            continue
        except EOFError:
            continue

def compute_melgram(audio_path, SR=12000, N_FFT=512, N_MELS=96, HOP_LEN=256, DURA=29.12):

    '''
    Compute a mel-spectrogram and returns it in a shape of (1,1,96,1366),
    where 96 == #mel-bins and 1366 == #time frame
    parameters
    Any format supported by audioread will work.
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


create_batch_from_id(sys.argv[1],None)
