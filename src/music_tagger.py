from argparse import ArgumentParser
import testing
import os


def printResults(predictions):

    from numpy import argsort

    labels = testing.label_words
    sort_indices = argsort(predictions)
    print("(" + "class" + ", " + "prediction prob" + ", " + "truth" + ")")
    for ind in sort_indices:
        print("(" + labels[ind] + ", " + str(predictions[ind] * 100)[:4] + "%)")


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('-m', dest='model', help='Provide model name from saved_model directory: CNN,CNN2,CNN3', required=True)
    parser.add_argument('-d', dest='data_id', help='provide any of the following ids: f, e, d, c, 1,2,3,4,5,6,7,8,9,0', metavar='DATA_ID')
    parser.add_argument('-a', dest='audiofile', help='path to audio file', metavar='AUDIO_FILE')

    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()
    print(options)
    if not os.path.isfile('/home/ubuntu/dla/DLA/saved_model/' + options.model + '.h5'):
        print('~/dla/DLA/saved_model/' + options.model + '.h5')
        parser.error("Invalid Model")

    if options.audiofile is not None:
        model, melgram, predictions = testing.testOnNew(options.model, options.audiofile)
        print("Printing results for " + options.audiofile.split('.')[0].split('/')[-1])
        printResults(predictions[0])

    elif options.data_id is not None:
        options.data_id in ['f', 'e', 'd', 'c', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        results = testing.testOnBatch(options.model, [options.data_id])
        testing.plotTopKROC(results, 5, 0.1)

    else:
        model, melgram, predictions = testing.testOnNew(options.model, '/home/ubuntu/dla/DLA/music_samples/MagnaSample/Sparks.mp3')
        print("Printing results for Sparks")
        printResults(predictions[0])

if __name__ == '__main__':
    main()