from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import datetime, os
import models
import numpy as np
from keras.models import load_model
lookup = np.array(['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go','silence','unknown'])

targets = ['bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go',
       'happy', 'house', 'left', 'marvin', 'nine', 'no', 'off', 'on',
       'one', 'right', 'seven', 'sheila', 'six', 'stop',
       'three', 'tree', 'two', 'up', 'wow', 'yes', 'zero','silence','other']

def transform_y(y):
    new_y = np.zeros((len(y),len(lookup)))
    for i, y_i in enumerate(y):
        label = targets[np.argmax(y_i)]
        if label in lookup:
            new_idx = np.where(lookup == label)[0]
            new_y[i,new_idx] = 1
        else:
            new_y[i,-1] = 1
    return new_y


input_dim= 16000
input_dim_spec = (256,32)
output_dim = len(lookup)

model = models.encoder(input_dim=input_dim, output_dim=output_dim, n_convs=4)
model = load_model('logs_20171214_154814\model.15-0.69.hdf5')
# optimizer = Adam(lr=0.0001)
# model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['categorical_accuracy'])
# model.summary()

logdir = 'logs_{}'.format(datetime.datetime.today().strftime('%Y%m%d_%H%M%S'))
os.mkdir(logdir)

callbacks = [
    EarlyStopping(monitor='val_loss',patience=50),
    TensorBoard(log_dir=logdir),
    ModelCheckpoint(filepath=os.path.join(logdir,'model.{epoch:02d}-{val_loss:.2f}.hdf5'),save_best_only=True)
]

from keras.utils import HDF5Matrix
from h5py import File

f = File('data2.h5')

x_train = HDF5Matrix('data2.h5', 'train/X', start=0, end=None, normalizer=None)
x_train =  f['train/X']
y_train = transform_y(f['train/y'][:,:31])

x_test = HDF5Matrix('data2.h5', 'test/X', start=0, end=None, normalizer=None)
y_test = transform_y(f['test/y'][:,:31])

model.fit(x=[x_train],
          y=y_train,
          batch_size=128,
          epochs=5000,
          validation_data=([x_test], y_test),
          callbacks=callbacks,
          shuffle="batch",
          initial_epoch=16
          )
