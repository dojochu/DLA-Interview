from audio_processor import create_batch_from_id, compute_melgram, get_clips_from_id
from keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as pyp


output_path = '/home/ubuntu/dla/DLA/saved_model/'
clip_path = '/media/ubuntu/9C33-6BBD/clip_info_final.csv'
label_path = '/media/ubuntu/9C33-6BBD/annotations_final.csv'
data_ids = ['f', 'e', 'd', 'c']

label_words = pd.read_csv(label_path, delimiter='\t').columns.tolist()[1:-1]

#Default parameters
n_mels = 96
sr = 12000
n_fft = 512
dura = 29.12
hop_len = 256


def getModel(model):

    '''
    Function: Retrieve the saved CNN model

    Arguments

        model: CNN model (CNN, CNN2, CNN3)

    Return: Model() for Keras library

    '''

    return load_model(output_path + model + '.h5')


def testOnNew(model, audio_path, label=None, SR=sr, N_FFT=n_fft, HOP_LEN=hop_len, DURA=dura, N_MELS=n_mels):

    '''
    Function: Run model on new audio file

    Arguments

        model: CNN model (CNN, CNN2, CNN3)
        audio_path: path of the audio file
        label: label for the audio file

    Return: Model() for Keras Library, Mel-spectrogram of audio file, model prediction

    '''

    trained_model = getModel(model)
    melgram = compute_melgram(audio_path, SR=SR, N_FFT=N_FFT, HOP_LEN=HOP_LEN, DURA=DURA, N_MELS=N_MELS)
    predictions = trained_model.predict(melgram)
    return trained_model, melgram, predictions


def testOnBatch(model, data_id=data_ids, sample=0, use_cache=True, SR=sr, N_FFT=n_fft, HOP_LEN=hop_len, DURA=dura, N_MELS=n_mels):

    '''
    Function: Run model on batch test data

    Arguments

        model: CNN model(CNN, CNN2, CNN3)
        data_id: ids from the batch data (f,e,d,c,0,1,2,3,4,5,6,7,8,9)
        sample: limit the sample
        use_cache: use cached data so you don't have to recompute mel-spectrogram

    Return: Model() for Keras library, Mel-spectrogram of audio file, evaluation label, model prediction, model metrics

    '''
    trained_model = getModel(model)
    if sample > 0:
        melgrams, labels = create_batch_from_id(get_clips_from_id(data_id)[:sample], use_cache, SR=SR, N_FFT=N_FFT, HOP_LEN=HOP_LEN, DURA=DURA, N_MELS=N_MELS)
    else:
        melgrams, labels = create_batch_from_id(get_clips_from_id(data_id), use_cache, SR=SR, N_FFT=N_FFT, HOP_LEN=HOP_LEN, DURA=DURA, N_MELS=N_MELS)

    results = trained_model.evaluate(melgrams, labels, batch_size=5)
    predictions = trained_model.predict(melgrams, batch_size=5)
    return trained_model, melgrams, labels, predictions, results


def plotResult(results, index, top):

    '''
    Function: Visually compare results of audio file to the evaluation label

    Arguments:

        results: batch output from testOnBatch
        index: index of the batch data
        top: your choice of K in the top K probabilities

    Return: Pandas dataframe with comparison

    '''

    label = results[2][index]
    pred = results[3][index]

    top = np.flipud(pred.argsort())[:top]

    pyp.figure(1)
    pyp.subplot(211)
    pyp.title('Evaluation Label')
    evaluation = [1 if a > 0 else 0 for a in label]
    pyp.plot(evaluation, color='b')
    pyp.subplot(212)
    pyp.title('Prediction')
    predictions = [1 if a in top else 0 for a in np.arange(0, len(pred))]
    pyp.plot(predictions, color='r')
    pyp.show()
    d = {'label': evaluation, 'prediction': predictions}
    df = pd.DataFrame(data=d, index=label_words)
    return df


def topKVec(results, index, top=10):

    '''
    Function: Compute vector of indices where the model correctly included label class in top K

    Arguments

        results: batch output from testOnBatch
        index: index of the batch data
        top: your choice of K in the top K probabilities

    Return: ones - list of indices 1 occurs in evaluation label vector, contain - vector same size as ones where label is contained in top K probabilities

    '''

    label = results[2][index]
    pred = results[3][index]

    ones = label.nonzero()[0]
    top = np.flipud(pred.argsort())[:top]

    contain = [1 if a in top else 0 for a in ones]

    return contain, ones

def topKMetric(results, top=10, cutoff=0):

    '''
    Function: Compute the top K Metric
    Arguments

        results: batch output from testOnBatch
        top: your choice of K in the top K probabilities
        cutoff: % criteria that is required in the top K probabilities to be considered a success

    Return: Top K metric, pandas dataframe showing correction and miss rate of each label class

    '''

    topK = 0.0
    total = 0.0
    missed = np.repeat(0,len(label_words)).tolist()
    correct = np.repeat(0, len(label_words)).tolist()

    for ind in range(0, results[2].shape[0]):
        contain, ones = topKVec(results, ind, top)
        for b in range(0, len(contain)):
            if contain[b] == 0:
                missed[ones[b]] += 1
            else:
                correct[ones[b]] += 1

        if np.sum(contain)/len(contain) >= cutoff:
           topK += 1.0
        if len(contain) > 0:
            total += 1.0

    misRate = pd.DataFrame(data={'correct':correct, 'missed':missed}, index=label_words)
    return topK / total, misRate

def plotTopKROC(results, top_steps = 10, cutoff_steps = 0.1):

    '''
    Function: Plot TopK ROC curve
    Arguments

        results: batch output from testOnBatch
        top_steps: number of steps for the top K
        cutoff_steps: number of steps for the % criteria

    Return: pyplot function

    '''

    NUM_LABELS = results[2].shape[1]
    top = np.arange(1, NUM_LABELS, top_steps)
    cutoff = np.arange(0, 1, cutoff_steps)

    pyp.figure(2)
    pyp.subplot(111)
    pyp.title('Top K Evaluation Curve')
    pyp.xlabel('K')
    pyp.ylabel('Metric')

    color_ind = 0
    legend = []
    for c  in cutoff:
        metric = []
        for t in top:
            metric.append(topKMetric(results, t, c)[0])
        legend.append(pyp.plot(top, metric, label='cutoff ' + str(c))[0])
        color_ind += 1
    pyp.legend(handles = legend)
    pyp.show()
    return pyp
