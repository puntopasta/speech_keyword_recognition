from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import datetime, os
import models


input_dim_raw = 16000
input_dim_spec = (256,32)
output_dim = 31

model = models.combined_bidirectional_conv_rnn(input_dim_raw=input_dim_raw, input_dim_spec = input_dim_spec, output_dim=output_dim)

optimizer = Adam(lr=0.0001)
model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['categorical_accuracy'])
model.summary()

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
x_spec_train = HDF5Matrix('data2.h5', 'train/X_spec', start=0, end=None, normalizer=None)
y_train = f['train/y'][:,:31]

x_test = HDF5Matrix('data2.h5', 'test/X', start=0, end=None, normalizer=None)
x_spec_test = HDF5Matrix('data2.h5', 'test/X_spec', start=0, end=None, normalizer=None)
y_test = f['test/y'][:,:31]

model.fit(x=[x_train, x_spec_train],
          y=y_train,
          epochs=5000,
          batch_size=64,
          validation_data=([x_test, x_spec_test], y_test),
          callbacks=callbacks,
          shuffle="batch"
          )
