# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 15:02:12 2017

@author: rvsloun
"""

import matplotlib.pyplot as plt
from data_generator import GeneratorFactory
from utils import getdirs
import numpy as np
from tqdm import tqdm, trange
import keras as K
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, LeakyReLU, BatchNormalization, Conv1D, MaxPool1D, AveragePooling1D, UpSampling1D, Flatten, Input, concatenate, Dropout, LSTM, Bidirectional
from keras.optimizers import SGD, Adam, RMSprop
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import time, os, datetime


def wrapper_gen(gen, adv=False):
    while True:
        d = next(gen)
        yield (d[0], d[2]) if adv else (d[0], d[1])

def encoder(input, n_convs):
    last_layer = input
    for i in range(n_convs):
        stride = 2
        conv = Conv1D(filters=64, kernel_size=5, strides=stride, name='Conv1D_{}'.format(i))(last_layer)
        conv = LeakyReLU()(conv)
        conv = BatchNormalization()(conv)
        last_layer = conv
    return last_layer, Model(inputs=input, outputs=last_layer)


def classifier_model(input_shape, n_target_classes):
    classifier = Sequential()
    classifier.add(LSTM(units=64, recurrent_dropout=0.5,input_shape=(input_shape[1],input_shape[2])))
    classifier.add(Dense(n_target_classes,activation='softmax'))
    return classifier


def encoder_classifier(input, e, c):
    m = Model(inputs=input, outputs=c(e))
    for l in m.layers:
        l.trainable = True
    return m


def loss_E_adverarial(y_true, y_pred):
    '''
    Loss function to maximize encoder error
    :return: 3 - categorical_crossentropy(pred - true)
    '''
    val = 3
    negativeCE = val - K.losses.categorical_crossentropy(y_pred, y_true)
    return negativeCE


def encoder_adversarial(input, encoder, adversarial):
    # compile model that maximizes error on encoder
    adversarial.trainable = False
    for l in adversarial.layers:
        l.trainable = False
    m = Model(inputs=input, outputs=adversarial(encoder))
    for l in m.layers[:-1]:
        l.trainable = True
    return m


def encoder_adversarial_train_adv(input, encoder, adversarial):
    # here, the adversarial will be trained
    adversarial.trainable = True
    for l in adversarial.layers:
        l.trainable = True
    m = Model(inputs=input, outputs=adversarial(encoder))
    for l in m.layers[:-1]:
         l.trainable = False
    return m


def plot_testing_accuracy(log):
    ax = plt.gca()
    ax.plot(10 * np.arange(0, len(log.E_C_loss.values)), log.E_C_loss.values, '-', color=(0, 0, 0), antialiased=True,
            label='Classifier loss')

    ax2 = ax.twinx()
    ax2.set_ylim([0, 1])
    ax2.plot(10 * np.arange(0, len(log.test_acc.values)), log.test_acc.values, '-',
             color=(164 / 255, 63 / 255, 152 / 255), antialiased=True, label='Classifier test accuracy')

    ax.set_xlabel('Iteration #')
    ax.set_ylabel('Loss', size=14)
    ax2.set_ylabel('Test accuracy', color=(164 / 255, 63 / 255, 152 / 255), size=14)
    ax2.set_ylim(0.4, 1)


def plot_adversarial_progress(log):
    plt.figure(figsize=(5, 5))
    ax = plt.gca()
    E_A_connect = [val for pair in zip(log.E_A_loss.values, log.E_A_loss_Eupdate.values) for val in pair]

    ax.plot(5 * np.arange(0, len(E_A_connect)), E_A_connect, '-', color=(0.8, 0.8, 0.8), antialiased=True)
    ax.plot(10 * np.arange(0, len(log.E_A_loss.values)), log.E_A_loss.values, 's',
            color=(72 / 255, 152 / 255, 211 / 255), antialiased=True, label='Adversarial update')
    ax.plot(10 * np.arange(0, len(log.E_A_loss_Eupdate.values)) + 5, log.E_A_loss_Eupdate.values, 'o',
            color=(164 / 255, 63 / 255, 152 / 255), antialiased=True, label='Encoder update')

    ax.set_xlabel('Iteration #')
    ax.set_ylabel('Loss', size=14)
    handles, labels = ax.get_legend_handles_labels()
    lg = ax.legend(handles, labels)
    lg.draw_frame(False)

    plt.show
    plt.pause(0.05)


def print_model_trainable(model):
    data = pd.DataFrame(data={
        'name' : [l.name for l in model.layers],
        'trainable': [l.trainable for l in model.layers],
        'output_shape': [l.output_shape for l in model.layers],
        'weights': [np.sum([np.prod(w.shape) for w in l.weights]) for l in model.layers],
        'trainable_weights': [np.sum([np.prod(w.shape) for w in l.trainable_weights]) for l in model.layers],
    }).set_index('name')
    print(data)
    print('Weights: ', data.weights.astype('int').sum())
    print('Trainable weights: ', data.trainable_weights.astype('int').sum())


def train(n_epochs):
    batch_size = 256
    improvement_patience = 100
    data_directory = getdirs()['data']
    factory = GeneratorFactory(other_class_rate=0.09, silence_class_rate=0.09, data_dir=data_directory, return_speaker=True)
    print('Loading Data... \n')
    train_gen, train_batches = factory.data_generator_train(batch_size = batch_size)
    test_gen, test_batches = factory.data_generator_test(batch_size = batch_size)
    X_test, Y_test, L_test = next(test_gen)

    input = Input(shape=(X_test.shape[1], 1))

    E_graph, E_model = encoder(input, n_convs=10)  # get both graph and model. Graph can be fed as input to other models
    print('Latent space shape: ', E_model.output_shape)
    C = classifier_model(input_shape=E_model.output_shape, n_target_classes=Y_test.shape[1])
    A = classifier_model(input_shape=E_model.output_shape, n_target_classes=L_test.shape[1])

    E_C = encoder_classifier(input, E_graph, C)
    E_C.compile(loss='categorical_crossentropy', optimizer=RMSprop(0.0001), metrics=['categorical_accuracy'])
    print('Classifier after encoder:\n')
    print_model_trainable(E_C)

    E_A_trainE = encoder_adversarial(input, E_graph, A)
    E_A_trainE.compile(loss=loss_E_adverarial, optimizer=RMSprop(0.0001),metrics=['categorical_accuracy'])
    print('Adversarial after encoder, train encoder:\n')
    print_model_trainable(E_A_trainE)

    E_A_trainA = encoder_adversarial_train_adv(input, E_graph,A)
    E_A_trainA.compile(loss='categorical_crossentropy', optimizer=RMSprop(0.0001),metrics=['categorical_accuracy'])
    print('Adversarial after encoder, train adversarial:\n')
    print_model_trainable(E_A_trainA)

    logdir = 'logs_can_{}'.format(datetime.datetime.today().strftime('%Y%m%d_%H%M%S'))
    os.mkdir(logdir)
    log = pd.DataFrame(columns=['E_A_loss', 'E_C_loss', 'E_A_loss_Eupdate', 'train_A_acc','train_C_acc',  'epoch', 'batch']).set_index(['epoch','batch'])
    log_test = pd.DataFrame(columns=['test_acc', 'test_loss', 'test_adv_loss', 'test_adv_acc', 'epoch', 'batch']).set_index(['epoch','batch'])

    for epoch in range(n_epochs):
        batch_looper = tqdm(range(int(train_batches)), total=train_batches, unit='batch')
        for i in batch_looper:
            start_time = time.time()
            X_train, Y_train, L_train = next(train_gen)
            data_load_time = (time.time() - start_time)

            E_C_loss, E_C_acc = E_C.train_on_batch(X_train, Y_train)  # train classifier:
            E_A_loss, E_A_acc = E_A_trainA.train_on_batch(X_train, L_train) # train adversarial given encoder:
            E_A_trainE.train_on_batch(X_train, L_train) # train encoder given adversarial:
            E_A_loss_Eupdate, tmp = E_A_trainA.evaluate(X_train, L_train, X_train.shape[0], verbose=0)

            log = log.append(pd.DataFrame(data={'E_C_loss': E_C_loss,
                                                'E_A_loss': E_A_loss,
                                                'E_A_loss_Eupdate': E_A_loss_Eupdate,
                                                'train_C_acc': E_C_acc,
                                                'train_A_acc': E_A_acc,
                                                'epoch': epoch,
                                                'batch': i}, index=[0]).set_index(['epoch', 'batch']))
            log.to_csv(os.path.join(logdir, 'training_log.csv'))

            batch_looper.desc = "[Epoch %i] train_loss = %0.4f \t train_acc = %0.4f \t adv loss a <> e = %0.4f <> %0.4f \t Data read: %2.2f" % (epoch, E_C_loss, E_C_acc, E_A_loss, E_A_loss_Eupdate, data_load_time)



        [test_loss, test_acc] = E_C.evaluate_generator(wrapper_gen(test_gen), steps=test_batches)
        [test_adv_loss, test_adv_acc] = E_A_trainA.evaluate_generator(wrapper_gen(test_gen,adv=True), steps=test_batches)

        if (len(log_test) == 0) or (log_test.tail(1)['test_loss'].values > test_loss):
            last_save = epoch
            E_model.save_weights(os.path.join(logdir, 'encoder'), True)
            C.save_weights(os.path.join(logdir,'classifier'), True)
            A.save_weights(os.path.join(logdir, 'adversarial'), True)
            E_C.save(filepath=os.path.join(logdir,'model_e{e}_loss_{l}_acc{a}'.format(e=epoch, l=E_C_loss, a = E_C_acc)))

        log_test = log_test.append(pd.DataFrame(data={'test_acc': test_acc,
                                            'test_loss': test_loss,
                                            'test_adv_loss' : test_adv_loss,
                                            'test_adv_acc' : test_adv_acc,
                                            'epoch': epoch,
                                            'batch': i}, index=[0]).set_index(['epoch', 'batch']))
        log_test.to_csv(os.path.join(logdir, 'test_log.csv'))

        if epoch > (last_save + improvement_patience):
            break


if __name__ == '__main__':
    # %% Train
    batch_size = 64
    train(n_epochs=50)

    # %% Infer
    # result = infer(folder)
