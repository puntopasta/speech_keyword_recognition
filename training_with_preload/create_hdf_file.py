from data_generator import GeneratorFactory
from utils import normalize, augment
import h5py

targets = ['bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go',
       'happy', 'house', 'left', 'marvin', 'nine', 'no', 'off', 'on',
       'one', 'right', 'seven', 'sheila', 'six', 'stop',
       'three', 'tree', 'two', 'up', 'wow', 'yes', 'zero']


batch_size = 128
factory = GeneratorFactory(batch_normalizer=normalize, classes=targets, augmentor=augment, progress=False, other_class_rate=0, silence_class_rate=0.03, online=True, data_dir ='../data', spectrogram='combined')


def save_to_file(grp, gen, bs, batches):
    X = grp.create_dataset('X', (bs*batches, 16000, 1))
    X_spec = grp.create_dataset('X_spec', (bs*batches, 256, 32, 1))
    y = grp.create_dataset('y', (bs*batches, 32))
    last_batch_idx = 0
    for i in range(int(batches)):
        print(i)
        x_batch, y_batch = next(gen)
        X[last_batch_idx:last_batch_idx+len(x_batch[0])] = x_batch[0]
        X_spec[last_batch_idx:last_batch_idx+len(x_batch[1])] = x_batch[1]
        y[last_batch_idx:last_batch_idx+len(y_batch)] = y_batch
        last_batch_idx = last_batch_idx + len(y_batch)


try:
    f = h5py.File('data2.h5')

    print('Generating train...')
    train_gen, train_batches = factory.data_generator_train(batch_size=batch_size)
    train_grp = f.create_group('train')
    save_to_file(train_grp,train_gen,batch_size, train_batches)

    print('Generating test...')
    test_gen, test_batches = factory.data_generator_test(batch_size=batch_size)
    test_grp = f.create_group('test')
    save_to_file(test_grp,test_gen,batch_size, test_batches)

    print('Generating validation...')
    val_gen, val_batches = factory.data_generator_validation(batch_size=batch_size)
    val_grp = f.create_group('validation')
    save_to_file(val_grp,val_gen,batch_size, val_batches)
finally:
    f.close()