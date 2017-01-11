# DLA Interview Project
This README documents the development of an automatic music tagger - a machine learning model that uses deep neural networks to categorize music files into a genre (rock, electronic, hip/hop, etc). My attempt is to experiment, train and test different architectures of deep learning models and analyze/visualize the results using TensorBoard. This project is being submitted as an interview for Deep Learning Analytics. To access this README in a more readable form, type the following in the terminal:

```
grip -b /home/ubuntu/dla/DLA/README.md 8080
```

##Author 
* Name: Stephanie Kao
* Email: stephanie.kao5@gmail.com
* Phone: 775-544-9655

## Problem Scope
Many music applications and recommendation engines heavily rely on tags to identify music files. One of the tags include music genre which is a strong indentifier for people's musical interest. Given a music audio file in the form of signals (i.e. frequency and amplitude of the signal over time), the key question is whether our machine learning model can accurately classify the musical genre of the song. The ground dataset is complicated in several ways:

1. Multi-categorical Songs: Songs can fall into more than one category.
2. Category Set: People classify songs differently and subjectively. Is it possible for a model to identify unique patterns in the audio signals that partitions the dataset into distinct genres? How do we define the initial genre set?

In the interest of time, I will use Keunwoo Choi's music genre tags defined below:

```
['rock', 'pop', 'alternative', 'indie', 'electronic', 'female vocalists', 
'dance', '00s', 'alternative rock', 'jazz', 'beautiful', 'metal', 
'chillout', 'male vocalists', 'classic rock', 'soul', 'indie rock',
'Mellow', 'electronica', '80s', 'folk', '90s', 'chill', 'instrumental',
'punk', 'oldies', 'blues', 'hard rock', 'ambient', 'acoustic', 'experimental',
'female vocalist', 'guitar', 'Hip-Hop', '70s', 'party', 'country', 'easy listening',
'sexy', 'catchy', 'funk', 'electro' ,'heavy metal', 'Progressive rock',
'60s', 'rnb', 'indie pop', 'sad', 'House', 'happy']
```

## Getting Started

These instructions will get the project up and running on this machine. 

The model was built in a python virtual environment using virutalenv. Activate the environment typing the following command in the terminal:

```
source /home/ubuntu/dla/bin/activate
```

All the necessary python packages required to run the model should be available in this environment. Type 'deactivate' to close the environment.

To tag a music file, run the music_tagger.py as follows:

```
python music_tagger.py -i MUSIC_FILE_PATH
```

## Files

* music_tagger.py - main program file
* audio_processor.py - import from Keunwoo Choi to read audio music files and convert any files to .wav, if necessary
* 

## Software

The music tagger model was trained and developed in Python using Tensorflow. Tensorflow was chosen over other Deep Learning tools (Caffe, Theano, Torch), because it was created in the Python language which closely integrates with numpy data manipulation tools and allows us access to a rich set of audio processing tools from the Python package - librosa. Since the Jetson TX1 Module already has Cuda toolket 8.0 and cuDNNv5 installed, Tensorflow was a natural choice to utilize the GPU.

# Data

The data chosen to complete the task is the well-known [MILLION SONG DATASET](http://labrosa.ee.columbia.edu/millionsong/). The music genre tags were obtained from [Last.fm Dataset](http://labrosa.ee.columbia.edu/millionsong/lastfm). 

## References

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.


## Acknowledgments

* Hat tip to anyone who's code was used
* Inspiration
* etc
