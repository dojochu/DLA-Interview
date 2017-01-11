from audio_processor import compute_melgram
import pandas as pd
import numpy as np
from argparse import ArgumentParser
import math
import os
from keras.models import Model, load_model

def build_parser():
    parser = ArgumentParser()
    parse.add_argument('--data-id', dest='data_id', help='data ids are any of these: f, e, d, c, 1,2,3,4,5,6,7,8,9,0', metavar='DATA_ID')
    parser.add_argument('--audiofile', dest='audiofile', help='audio data file', metavar='AUDIO_FILE')
    parser.add_argument('--model', dest='model', help='Provide model name from saved_model directory', required=True)
    parser.add_argument('--data-size', dest='data_size', help='if data id provided need to provide data size', metavar='DATA_SIZE')
 	return parser

def main():
    parser = build_parser()
    options = parser.parse_args()

    #input and output
    input_file = options.audiofile
    model_file = '/home/ubuntu/dla/DLA/output/saved_model/' + options.model + '.h5'

    try:
        model = load_model(model_file)
    except FileNotFoundError
	print('Couldnt not find the model specified')	
    
    if options.data_id is in ['f', 'e', 'd', 'c', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
        melgrams, labels = create_batch_from_id(options.data_id, options.data_size)
	print(model.evaluate(melgrams, labels))
