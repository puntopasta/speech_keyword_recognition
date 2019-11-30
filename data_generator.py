import os, glob
from scipy.io import wavfile
import numpy as np
import pandas as pd
import utils
from utils import getdirs, augment_spectogram

class GeneratorFactory:

    def __init__(self, data_dir =getdirs()['data'],
    classes = np.array(['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']),
    fs=16000, other_class_rate = 0.09, silence_class_rate = 0.09, spectrogram = 'no', return_speaker = False, parallel = True):
        '''
        Generator factory. Can be used to create generators that will load data for training neural nets.
        Remove the background_noise folder from the data folder. Otherwise it will mess up.
        If you want silence then please run create_silent_samples.py first which will create silence class used here. Then remove background_noise folder.
        :param data_dir: Data dir (should contain a train and test subfolder)
        :param classes: array of string representation of classes. Default is the 10 challenge classes. If none is passed all classes are used.
        :param fs: fs of signal
        :param other_class_rate: how much of each dataset should be populated by "other" labels. (fraction of dataset size), default 0.2
        :param silence_class_rate: how much of the silence class should be populated
        :param spectrogram: returns spectrograms "yes" for only spectrogram, "combined" for both raw data and spectrogram, "no" (default) for only raw data
        :param return_speaker: returns speaker ID if True (default=False)
        '''

        self.data_dir = data_dir
        self.fs = fs
        self.return_speaker = return_speaker
        self.silence_class_rate = silence_class_rate
        self.other_class_rate = other_class_rate
        self.spectrogram = spectrogram
        self.parallel = parallel

        all_files = [st[st.find('audio')+6:] for st in glob.glob(os.path.join(data_dir,'train','audio','*','*.wav'))]
        self.testing_list = pd.read_csv(os.path.join(data_dir,'train', 'testing_list.txt')).astype('str').as_matrix().squeeze()
        self.validation_list = pd.read_csv(os.path.join(data_dir,'train', 'validation_list.txt')).astype('str').as_matrix().squeeze()
        self.training_list = np.array([f for f in all_files if (f not in self.validation_list) and (f not in self.testing_list)])
        self.speakers = np.unique([f[f.find(os.path.sep) + len(os.path.sep):f.find('_')] for f in self.training_list])
        self.classes = np.array(classes) if classes is not None else np.unique([file[:file.find(os.sep)] for file in all_files])

        # filter based on relevant classes
        self.training_silence = [f[f.find('audio')+6:] for f in glob.glob(os.path.join(data_dir,'train','audio','silence','train_*.wav'))]
        self.training_other = [file for file in self.training_list if self.get_class(file) not in self.classes]
        self.training_list = [file for file in self.training_list if self.get_class(file) in self.classes]

        self.testing_silence = [f[f.find('audio')+6:] for f in glob.glob(os.path.join(data_dir,'train','audio','silence','test_*.wav'))]
        self.testing_other = [file for file in self.testing_list if self.get_class(file) not in self.classes]
        self.testing_list = [file for file in self.testing_list if self.get_class(file) in self.classes]

        self.validation_silence = [f[f.find('audio')+6:] for f in glob.glob(os.path.join(data_dir,'train','audio','silence','validation_*.wav'))]
        self.validation_other = [file for file in self.validation_list if self.get_class(file) not in self.classes]
        self.validation_list = [file for file in self.validation_list if self.get_class(file) in self.classes]


    def get_class(self, file):
        return os.path.split(file)[0]

    def extract_signal(self, file):
        fs = 16000
        (category, filename) = os.path.split(file)
        if category in self.classes:
            class_index = np.where(category == self.classes)[0]
        elif category == 'silence':
            class_index = len(self.classes) # silence class
        else:
            class_index = len(self.classes)+1 # other class

        target = np.zeros((1,len(self.classes) + 2),dtype='int16')
        target[0,class_index] = 1
        waveform = wavfile.read(os.path.join(self.data_dir,'train','audio',file))[1]
        waveform = np.concatenate([waveform, np.zeros((fs - len(waveform)), dtype='int16')])
        waveform = np.expand_dims(waveform, 0 )
        subject_name = filename[:filename.find('_')]
        speaker_onehot = np.zeros((1,len(self.speakers)))
        speaker_onehot[0,np.where(self.speakers == subject_name)[0]] = 1
        return waveform, category, speaker_onehot, target


    def extract_all_signals(self, file_list):
        X = []
        y = []
        L = []
        class_name = []

        for ix, file in enumerate(file_list):
            waveform, category, subject_name, target = self.extract_signal(file)
            X.append(waveform)
            y.append(target)
            L.append(subject_name)
            class_name.append(category)

        return np.concatenate(X), \
               np.concatenate(y), \
               np.concatenate(L), \
               np.array(class_name)


    def holdout_generator(self, batch_size):
        holdout_files = glob.glob(os.path.join(self.data_dir,'test','audio','*.wav'))
        if batch_size is None:
            batch_size = len(holdout_files)

        for i in np.arange(0,len(holdout_files)+batch_size, batch_size):
            batch_files = holdout_files[i:np.min([i+batch_size,len(holdout_files)])]
            if len(batch_files) == 0:
                return
            X = []
            for file in batch_files:
                waveform = wavfile.read(file)[1]
                waveform = np.concatenate([waveform, np.zeros((self.fs - len(waveform)))])
                waveform = np.expand_dims(waveform, 0 )
                X.append(waveform)

            X = np.concatenate(X)
            X = utils.normalize(X)


            if self.spectrogram != 'no':
                X_spec = utils.convert_to_spectrogram(X, fft_mode='mel')
                X_spec = np.expand_dims(X_spec,-1)
            X = np.expand_dims(X, -1)

            if self.spectrogram == 'yes':
                yield X_spec, batch_files
            if self.spectrogram == 'combined':
                yield [X, X_spec], batch_files
            if self.spectrogram == 'no':
                yield X, batch_files

    def _sample_from(self, file_list, size):
        if size == 0:
            return [], [], [], []

        files = np.random.choice(file_list, size=size, replace=True)
        return self.extract_all_signals(files)

    @staticmethod
    def post_process(inp):
        (X, y, augment, spectrogram) = inp

        X = utils.normalize(X)

        if augment:
            X = utils.augment(X)

        if spectrogram != 'no':
            X_spec = utils.convert_to_spectrogram(X)
            X_spec = augment_spectogram(X_spec)
            X_spec = np.expand_dims(X_spec, -1)

        X = np.expand_dims(X, -1)

        if spectrogram == 'yes':
            return X_spec, y
        elif spectrogram == 'combined':
            return [X, X_spec], y
        elif spectrogram == 'no':
            return X, y
        else:
            raise Exception(
                'Invalid spectrogram type: {}. Please select either yes, no or combined'.format(spectrogram))

    def _data_generator_online(self, file_list, other_list, silence_list, batch_size, augment):
        if batch_size is None:
            batch_size = len(file_list)

        randomized_order = np.arange(len(file_list))
        np.random.shuffle(randomized_order)

        other_class_to_sample = int(batch_size * self.other_class_rate)
        silence_class_to_sample = int(batch_size * self.silence_class_rate)

        batch_size = batch_size - other_class_to_sample - silence_class_to_sample
        from multiprocessing.pool import ThreadPool
        from multiprocessing import cpu_count
        n_cpus = cpu_count() - 1
        pool = ThreadPool(n_cpus)
        data_stack = []
        speaker_stack = []
        while True:
            for i in np.arange(0,len(file_list), batch_size):
                files = [file_list[j] for j in randomized_order[i:i+batch_size]]
                X_c, y_c, L_c, class_name = self.extract_all_signals(files)
                X_o, y_o, L_o, class_name_o = self._sample_from(other_list, other_class_to_sample)
                X_s, y_s, L_s, class_name_s = self._sample_from(silence_list, silence_class_to_sample)
                X = np.concatenate([x for x in [X_c, X_o, X_s] if len(x) > 0])
                y = np.concatenate([y for y in [y_c, y_o, y_s] if len(y) > 0])
                L= np.concatenate([l for l in [L_c, L_o, L_s] if len(l) > 0])
                reorder = np.arange(len(X))
                np.random.shuffle(reorder)
                X, y, L = X[reorder], y[reorder], L[reorder]
                if self.parallel:
                    data_stack.append((X,y, augment, self.spectrogram))
                    speaker_stack.append(L)
                    if (len(data_stack) == n_cpus) or ((i + batch_size) > len(file_list)):
                        post_processed = pool.map(self.post_process, data_stack)
                        for i, (X, y) in enumerate(post_processed):
                            yield (X, y, speaker_stack[i]) if self.return_speaker else (X, y)
                        data_stack = []
                        speaker_stack = []
                else:
                    X, y = self.post_process((X,y, augment, self.spectrogram))
                    yield (X, y, L) if self.return_speaker else (X, y)

    def data_generator(self, file_list, other_list, silence_list, batch_size, augment):
        total_files = len(file_list) + len(file_list)*(self.other_class_rate + self.silence_class_rate)
        n_batches = np.round(total_files/batch_size)
        return self._data_generator_online(file_list, other_list, silence_list, batch_size, augment), n_batches

    def data_generator_train(self, batch_size = None, augment = True):
        '''
        Returns generator for the train set.
        :param batch_size: batch size per generation. If none, entire dataset is returned in first generation.
        :return: generator object.
        '''
        return self.data_generator(self.training_list, self.training_other, self.training_silence, batch_size, augment)

    def data_generator_test(self, batch_size = None, augment = False):
        '''
        Returns generator for the test set.
        :param batch_size: batch size per generation. If none, entire dataset is returned in first generation.
        :return: generator object.
        '''
        return self.data_generator(self.testing_list, self.testing_other, self.testing_silence, batch_size, augment)

    def data_generator_validation(self, batch_size = None, augment = False):
        '''
        Returns generator for the validation set.
        :param batch_size: batch size per generation. If none, entire dataset is returned in first generation.
        :return: generator object.
        '''
        return self.data_generator(self.validation_list, self.validation_other, self.validation_silence, batch_size, augment)


    @classmethod
    def _data_generator_silence(cls, silence_list, spoken_list,batch_size= None,augment=False):
        gen = cls.data_generator(file_list=silence_list + spoken_list,
                                  other_list=None,
                                  silence_list=None,
                                  batch_size=batch_size,
                                  augment=augment)[0]
        for X, y in gen:
            yield X, np.expand_dims(y[:,-2], -1)


    def data_generator_silence_train(self, batch_size =None, augment = False):
        silence = self.training_silence
        spoken =self.training_list + self.training_other
        n_batches = np.round((len(silence) + len(spoken)) / batch_size)
        return n_batches, self._data_generator_silence(silence_list=silence,
                                                       spoken_list=spoken,
                                                       batch_size=batch_size,
                                                       augment=augment)

    def data_generator_silence_test(self, batch_size =None, augment = False):
        silence = self.testing_silence
        spoken = self.testing_list + self.testing_other
        n_batches = np.round((len(silence) + len(spoken)) / batch_size)
        return n_batches, self._data_generator_silence(silence_list=silence,
                                                       spoken_list=spoken,
                                                       batch_size=batch_size,
                                                       augment=augment)

    def data_generator_silence_validation(self, batch_size =None, augment = False):
        silence = self.validation_silence
        spoken = self.validation_other + self.validation_list
        n_batches = np.round((len(silence) + len(spoken)) / batch_size)
        return n_batches, self._data_generator_silence(silence_list=silence,
                                                       spoken_list=spoken,
                                                       batch_size=batch_size,
                                                       augment=augment)
