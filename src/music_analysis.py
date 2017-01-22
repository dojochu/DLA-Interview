import audio_processor as ap
import matplotlib.pyplot as pyp


def plot_spectrogram_grid(music):

    '''
    Function: Plot Mel-Spectrogram

    Arguments

        music: melgram or mel-spectrogram

    Return: figure and axes from pyplot

    '''

    fig, ax = pyp.subplots(2, int(len(music) / 2) if len(music)%2==0 else int(len(music)/2+1))
    for ind in range(0, len(music)):
        pl = ax.flat[ind].pcolormesh(music[ind][0])

    fig.subplots_adjust(right=0.8)
    cbar = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(pl, cax=cbar)

    pyp.show()
    return fig, ax


def sample(attributes=None, n=None):

    '''
    Function: Plot a sample of mel-spectrograms with particular label (i.e. violin, vocal, guitar, etc)
    Arguments

        attributes: class labels to sample
        n: limit the sample

    Return: NA

    '''
    melgram, label = ap.create_batch_from_id(ap.get_clips_from_label(attributes)[:n], True)

    plot_spectrogram_grid(melgram)
