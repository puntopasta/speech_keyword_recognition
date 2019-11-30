from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from data_generator import GeneratorFactory
from utils import getdirs
import datetime, os
import models
import pandas as pd




data_dir = getdirs()['data']
batch_size = 256
factory = GeneratorFactory(other_class_rate=0.09, silence_class_rate=0.09, data_dir =data_dir, return_speaker=False)

train_gen, train_batches = factory.data_generator_train(batch_size=batch_size)
test_gen, test_batches = factory.data_generator_test(batch_size=batch_size)
val_gen, val_batches = factory.data_generator_validation(batch_size=batch_size)

batch_x, batch_y = next(train_gen)
input_dim = batch_x.shape[1]
output_dim = batch_y.shape[1]

model = models.bidirectional_conv_rnn(input_dim=input_dim, output_dim=output_dim, n_convs=10)

optimizer = Adam(lr=0.001)
model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['categorical_accuracy'])
model.summary()

logdir = 'logs_{}'.format(datetime.datetime.today().strftime('%Y%m%d_%H%M%S'))
os.mkdir(logdir)

callbacks = [
    EarlyStopping(monitor='val_loss',patience=200),
    TensorBoard(log_dir=logdir),
    ModelCheckpoint(filepath=os.path.join(logdir,'model.{epoch:02d}-{val_loss:.2f}.hdf5'),save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
]

model.fit_generator(generator=train_gen,
                    steps_per_epoch=train_batches,
                    epochs=5000,
                    validation_data=test_gen,
                    validation_steps = test_batches,
                    callbacks=callbacks,
                    max_queue_size=40,
                    use_multiprocessing=False
                    )
