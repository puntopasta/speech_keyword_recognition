from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from data_generator import GeneratorFactory
from utils import normalize
import datetime, os
import trained_models

batch_size = 256
factory = GeneratorFactory(batch_normalizer=normalize, progress=False, other_class_rate=0.0, silence_class_rate=0.0,  online=True , data_dir='D:\Mustafa\workspace\data')

n_train, train_gen = factory.data_generator_silence_train(batch_size=batch_size)
n_test, test_gen = factory.data_generator_silence_test(batch_size=batch_size)
n_validation, validation_gen = factory.data_generator_silence_validation(batch_size=batch_size)

batch_x, batch_y = next(train_gen)
input_dim = batch_x.shape[1]
output_dim = batch_y.shape[1]

model = trained_models.conv_only(input_dim=input_dim, output_dim=output_dim, n_convs=10)

optimizer = Adam(lr=0.0001)
model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['binary_accuracy'])
model.summary()

logdir = 'logs_{}'.format(datetime.datetime.today().strftime('%Y%m%d_%H%M%S'))
os.mkdir(logdir)

callbacks = [
    EarlyStopping(monitor='val_loss',patience=50),
    TensorBoard(log_dir=logdir),
    ModelCheckpoint(filepath=os.path.join(logdir,'model.{epoch:02d}-{val_loss:.2f}.hdf5'),save_best_only=True)
]

model.fit_generator(generator=train_gen,
                    steps_per_epoch=n_train,
                    epochs=1,
                    validation_data=test_gen,
                    validation_steps = n_test,
                    callbacks=callbacks,
                    shuffle=True,
                    max_queue_size=10
                    )


