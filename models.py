from keras.models import Model
from keras.layers import Dense, Input, LSTM, BatchNormalization, Convolution1D, Dropout, Bidirectional, GRU, concatenate, AveragePooling1D, Flatten, Conv2D, MaxPooling2D
from keras.layers import LeakyReLU
import numpy as np


def first_try(input_dim, output_dim, n_convs):

    input = Input(shape=(input_dim,1))
    convs = []
    last_layer = input
    for i in range(n_convs):
        conv = Convolution1D(filters=64, kernel_size=5, strides=2, name='Conv1D_{}'.format(i))(last_layer)
        conv = LeakyReLU()(conv)
        conv = BatchNormalization()(conv)
        convs.append(conv)
        last_layer = conv
    lstm = Bidirectional(LSTM(units=256))(last_layer)
    out = Dense(output_dim,activation='softmax')(lstm)
    m = Model(inputs=input, outputs=out)
    return m


# train around 92. test around 88. validation around the same. Scores 0.60
def submission1(input_dim, output_dim, n_convs):
    input = Input(shape=(input_dim,1))
    convs = []
    last_layer = input
    for i in range(n_convs):
        conv = Convolution1D(filters=64, kernel_size=5, strides=2, name='Conv1D_{}'.format(i))(last_layer)
        conv = LeakyReLU()(conv)
        conv = BatchNormalization()(conv)
        convs.append(conv)
        last_layer = conv
    last_layer = Dropout(0.5)(last_layer)
    lstm = LSTM(units=256, recurrent_dropout=0.5)(last_layer)
    out = Dense(output_dim,activation='softmax')(lstm)
    m = Model(inputs=input, outputs=out)
    return m


def bidirectional_conv_rnn(input_dim, output_dim, n_convs):
    from keras.models import Model
    from keras.layers import Dense, Input, LSTM, BatchNormalization, Convolution1D, Dropout
    from keras.layers import LeakyReLU
    input = Input(shape=(input_dim,1))
    convs = []
    last_layer = input
    for i in range(n_convs):
        stride = 2
        conv = Convolution1D(filters=64, kernel_size=5, strides=stride, name='Conv1D_{}'.format(i))(last_layer)
        conv = LeakyReLU()(conv)
        conv = BatchNormalization()(conv)
        convs.append(conv)
        last_layer = conv
    last_layer = Dropout(0.5)(last_layer)
    lstm = Bidirectional(LSTM(units=256, recurrent_dropout=0.5))(last_layer)
    out = Dense(output_dim,activation='softmax')(lstm)
    m = Model(inputs=input, outputs=out)
    return m


def bidirectional_conv_moar_lstm(input_dim, output_dim, n_convs):
    from keras.models import Model
    from keras.layers import Dense, Input, LSTM, BatchNormalization, Convolution1D, Dropout
    from keras.layers import LeakyReLU
    input = Input(shape=(input_dim,1))
    convs = []
    last_layer = input
    for i in range(n_convs):
        stride = 2
        conv = Convolution1D(filters=64, kernel_size=5, strides=stride, name='Conv1D_{}'.format(i))(last_layer)
        conv = LeakyReLU()(conv)
        conv = BatchNormalization()(conv)
        convs.append(conv)
        last_layer = conv
    last_layer = Dropout(0.5)(last_layer)
    lstm = Bidirectional(LSTM(units=256, recurrent_dropout=0.5, return_sequences=True, dropout=0.5))(last_layer)
    lstm = Bidirectional(LSTM(units=256, recurrent_dropout=0.5, return_sequences=True, dropout=0.5))(lstm)
    lstm = Bidirectional(LSTM(units=256, recurrent_dropout=0.5, dropout=0.5))(lstm)
    out = Dense(output_dim,activation='softmax')(lstm)
    m = Model(inputs=input, outputs=out)
    return m


def bidirectional_conv_rnn_with_conv_dropout(input_dim, output_dim, n_convs):
    from keras.models import Model
    from keras.layers import Dense, Input, LSTM, BatchNormalization, Convolution1D, Dropout
    from keras.layers import LeakyReLU
    input = Input(shape=(input_dim,1))
    convs = []
    last_layer = input
    for i in range(n_convs):
        conv = Convolution1D(filters=64, kernel_size=5, strides=2, name='Conv1D_{}'.format(i))(last_layer)
        conv = LeakyReLU()(conv)
        conv = BatchNormalization()(conv)
        if np.mod(i,3) == 0 and (i+1) != n_convs:
            conv = Dropout(0.5)(conv)
        convs.append(conv)
        last_layer = conv
    last_layer = Dropout(0.5)(last_layer)
    lstm = Bidirectional(LSTM(units=256, recurrent_dropout=0.5))(last_layer)
    out = Dense(output_dim,activation='softmax')(lstm)
    m = Model(inputs=input, outputs=out)
    return m

def conv_only(input_dim, output_dim, n_convs):
    from keras.models import Model
    from keras.layers import Dense, Input, LSTM, BatchNormalization, Convolution1D, Dropout
    from keras.layers import LeakyReLU
    input = Input(shape=(input_dim,1))
    convs = []
    last_layer = input
    for i in range(n_convs):
        stride = 2   if np.mod(i,2) == 0 else 1
        conv = Convolution1D(filters=64, kernel_size=5, strides=stride,  name='Conv1D_{}'.format(i))(last_layer)
        conv = BatchNormalization()(conv)
        conv = LeakyReLU()(conv)
        if np.mod(i,2) != 0:
            conv = AveragePooling1D(2)(conv)
            conv = Dropout(0.5)(conv)

        convs.append(conv)
        last_layer = conv
    last_layer = Dropout(0.5)(last_layer)
    out = Flatten()(last_layer)
    out = Dense(256)(out)
    out = LeakyReLU()(out)
    out = Dense(128)(out)
    out = LeakyReLU()(out)
    out = Dense(64)(out)
    out = LeakyReLU()(out)
    out = Dense(output_dim,activation='softmax')(out)
    m = Model(inputs=input, outputs=out)
    return m


def bidirectional_conv_rnn__with_conv_and_input_dropout(input_dim, output_dim, n_convs):
    from keras.models import Model
    from keras.layers import Dense, Input, LSTM, BatchNormalization, Convolution1D, Dropout
    from keras.layers import LeakyReLU
    input = Input(shape=(input_dim,1))
    convs = []
    last_layer = input
    for i in range(n_convs):
        conv = Convolution1D(filters=64, kernel_size=5, strides=2, name='Conv1D_{}'.format(i))(last_layer)
        conv = LeakyReLU()(conv)
        conv = BatchNormalization()(conv)
        if np.mod(i,3) == 0 and (i+1) != n_convs:
            conv = Dropout(0.5)(conv)
        convs.append(conv)
        last_layer = conv
    last_layer = Dropout(0.5)(last_layer)
    lstm = Bidirectional(LSTM(units=256, recurrent_dropout=0.5))(last_layer)
    out = Dense(output_dim,activation='softmax')(lstm)
    m = Model(inputs=input, outputs=out)
    return m


def inception_block(input, stride=(2,2), filters=64):
    tower1 = Conv2D(filters=filters, shape=(1,1), padding='same')(input)
    tower1 = LeakyReLU()(tower1)
    tower1 = Conv2D(filters=filters, shape=(3, 3), strides=stride, padding='same')(tower1)
    tower1 = LeakyReLU()(tower1)

    tower2 = Conv2D(filters=filters, shape=(1, 1), padding='same')(input)
    tower2 = LeakyReLU()(tower2)
    tower2 = Conv2D(filters=filters, shape=(5, 5), strides=stride, padding='same')(tower2)
    tower2 = LeakyReLU()(tower2)
    tower3 = MaxPooling2D((3,3),strides=(1,1),padding='same')(input)
    tower3 = Conv2D(filters=filters, shape=(1, 1), strides=stride, padding='same')(tower3)
    tower3 = LeakyReLU()(tower3)

    return concatenate([tower1,tower2,tower3],axis=-1)


def spectrogram_inception_convs(input_dim, output_dim):
    from keras.models import Model
    from keras.layers import Dense, Input, LSTM, BatchNormalization, Convolution2D, Dropout, Bidirectional, Reshape
    input = Input(shape=(input_dim[0],input_dim[1],1))
    conv = BatchNormalization()(input)
    conv = inception_block(conv)
    conv = inception_block(conv)
    conv = inception_block(conv)
    conv = inception_block(conv)
    conv = Convolution2D(filters=16, kernel_size=(2, 5), strides=(1, 2))(conv)
    conv = Reshape((11,80))(conv)
    conv = Dropout(0.5)(conv)
    lstm = Bidirectional(LSTM(units=16, recurrent_dropout=0.5, name='LSTM'), name='LSTM_bidirectional')(conv)
    out = Dense(output_dim,activation='softmax')(lstm)
    m = Model(inputs=input, outputs=out)
    return m

def spectrogram_bidirectional_conv_rnn(input_dim, output_dim):
    from keras.models import Model
    from keras.layers import Dense, Input, LSTM, BatchNormalization, Convolution2D, Dropout, Bidirectional, LeakyReLU, Flatten, Permute, Reshape
    input = Input(shape=(input_dim[0],input_dim[1],1))
    conv = Convolution2D(filters=64, kernel_size=(5,5), strides=(2,2), name='Conv1')(input)
    conv = BatchNormalization()(conv)
    conv = LeakyReLU()(conv)
    conv = Convolution2D(filters=64, kernel_size=(5,5), strides=(2,2), name='Conv2')(conv)
    conv = BatchNormalization()(conv)
    conv = LeakyReLU()(conv)
    conv = Convolution2D(filters=64, kernel_size=(5,5), strides=(2,2), name='Conv3')(conv)
    conv = BatchNormalization()(conv)
    conv = LeakyReLU()(conv)
    conv = Convolution2D(filters=64, kernel_size=(5,5), strides=(2,2), name='Conv4')(conv)
    conv = BatchNormalization()(conv)
    conv = LeakyReLU()(conv)
    conv = Convolution2D(filters=16, kernel_size=(2,5), strides=(1,2), name='Conv5')(conv)
    conv = BatchNormalization()(conv)
    conv = LeakyReLU()(conv)
    conv = Reshape((11,80))(conv)
    conv = Dropout(0.5)(conv)
    lstm = Bidirectional(LSTM(units=16, recurrent_dropout=0.5, name='LSTM'), name='LSTM_bidirectional')(conv)
    out = Dense(output_dim,activation='softmax')(lstm)
    m = Model(inputs=input, outputs=out)
    return m


def combined_bidirectional_conv_rnn(input_dim_raw, input_dim_spec, output_dim):
    from keras.models import Model
    from keras.layers import Dense, Input, LSTM, BatchNormalization, Convolution1D, Convolution2D, Dropout, Bidirectional, LeakyReLU, Reshape, concatenate
    input_spec = Input(shape=(input_dim_spec[0],input_dim_spec[1],1))
    conv = Convolution2D(filters=128, kernel_size=(5,2), strides=(2,1), name='Conv1')(input_spec)
    conv = BatchNormalization()(conv)
    conv = LeakyReLU()(conv)
    conv = Convolution2D(filters=64, kernel_size=(5,2), strides=(2,1), name='Conv2')(conv)
    conv = BatchNormalization()(conv)
    conv = LeakyReLU()(conv)
    conv = Convolution2D(filters=32, kernel_size=(5,2), strides=(2,2), name='Conv3')(conv)
    conv = BatchNormalization()(conv)
    conv = LeakyReLU()(conv)
    conv = Convolution2D(filters=16, kernel_size=(5,2), strides=(2,2), name='Conv4')(conv)
    conv = BatchNormalization()(conv)
    conv = LeakyReLU()(conv)
    conv = Reshape((13,112))(conv)
    conv = Dropout(0.5)(conv)
    lstm1 = Bidirectional(LSTM(units=256, recurrent_dropout=0.5, name='LSTM'), name='LSTM_bidirectional')(conv)

    input_raw = Input(shape=(input_dim_raw,1))
    convs = []
    last_layer = input_raw
    for i in range(8):
        stride = 2
        conv = Convolution1D(filters=64, kernel_size=5, strides=stride, name='Conv1D_{}'.format(i))(last_layer)
        conv = LeakyReLU()(conv)
        conv = BatchNormalization()(conv)
        convs.append(conv)
        last_layer = conv
    last_layer = Dropout(0.5)(last_layer)
    lstm2 = Bidirectional(LSTM(units=256, recurrent_dropout=0.5, return_sequences=True, dropout=0.5))(last_layer)
    lstm2 = Bidirectional(LSTM(units=256, recurrent_dropout=0.5, return_sequences=True, dropout=0.5))(lstm2)
    lstm2 = Bidirectional(LSTM(units=256, recurrent_dropout=0.5, dropout=0.5))(lstm2)

    out = concatenate([lstm1, lstm2],axis=-1)
    out = Dense(64, activation='sigmoid')(out)
    out = Dense(output_dim,activation='softmax')(out)
    m = Model(inputs=[input_raw, input_spec], outputs=out)

    return m


def encoder(input_dim, output_dim, n_convs):
    from keras.models import Model
    from keras.layers import Dense, Input, LSTM, BatchNormalization, Convolution1D, MaxPool1D
    from keras.layers import LeakyReLU

    input = Input(shape=(input_dim,1))
    convs = []
    last_layer = input
    for i in range(n_convs):
        conv = Convolution1D(filters=64, kernel_size=5, strides=1, name='Conv1D_{}'.format(i))(last_layer)
        conv = LeakyReLU()(conv)
        conv = BatchNormalization()(conv)
        conv = MaxPool1D(4)(conv)
        convs.append(conv)
        last_layer = conv
    lstm = Bidirectional(LSTM(units=128, recurrent_dropout=0.5,  dropout=0.5))(last_layer)
    out = Dense(32, activation='sigmoid')(lstm)
    out = Dense(output_dim,activation='softmax')(out)
    return Model(input, out)
