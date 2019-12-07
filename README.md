# speech_keyword_recognition

Entry in Kaggle's Tensorflow keyword recognition challenge.

This was an entry for the Tensorflow speech recognition challenge on Kaggle: https://www.kaggle.com/c/tensorflow-speech-recognition-challenge

The goal was to build an algorithm that could detect common commands/keywords such as "start", "stop", etc. The entry reached an 87% accuracy on the holdout test set, classifying 10 different keyword commands, as well as the "silence" and "non-keyword" classes.

Includes a probabilistic data generation and augmentation pipeline as well as a number of machine learning models that were experimented with.
