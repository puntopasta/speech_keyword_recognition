from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from data_generator import GeneratorFactory
from utils import normalize, augment
import datetime, os
import models

batch_size = 128
factory = GeneratorFactory(batch_normalizer=normalize, augmentor=augment, progress=False, other_class_rate=0.09, silence_class_rate=0.09, online=True, data_dir ='../data', spectrogram='combined')
train_gen, train_batches = factory.data_generator_train(batch_size=batch_size)
test_gen, test_batches = factory.data_generator_test(batch_size=batch_size)
val_gen, val_batches = factory.data_generator_validation(batch_size=batch_size)

batch_x, batch_y = next(train_gen)
input_dim_wave = batch_x[0].shape[1]
input_dim_spec = batch_x[1].shape[1:]
output_dim = batch_y.shape[1]

model = models.combined_model(input_dim_spec=input_dim_spec, input_dim_wave = input_dim_wave, output_dim=output_dim)

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

model.fit_generator(generator=train_gen,
                    steps_per_epoch=train_batches,
                    epochs=5000,
                    validation_data=test_gen,
                    validation_steps = test_batches,
                    callbacks=callbacks,
                    shuffle=True,
                    max_queue_size=10,
                    use_multiprocessing=True,
                    workers=8
                    )
