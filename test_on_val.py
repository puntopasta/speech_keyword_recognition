from data_generator import GeneratorFactory
from utils import normalize
from utils import augment
from keras.models import load_model
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


data_directory = '../data'
batch_size = 256
factory = GeneratorFactory(batch_normalizer=normalize, augmentor = augment, progress=False, other_class_rate=0.09, silence_class_rate=0.09, online=True, data_dir = data_directory, spectrogram='no')


test_gen, test_batches = factory.data_generator_test(batch_size=batch_size)
val_gen, val_batches = factory.data_generator_validation(batch_size=batch_size)

model = load_model('logs_20171209_151639/model.303-0.53.hdf5')

print(model.evaluate_generator(generator=test_gen, steps=val_batches))
print(model.evaluate_generator(generator=val_gen, steps=val_batches))
