import numpy as np
import librosa, threading
from numpy.fft import irfft
from pysndfx import AudioEffectsChain
from python_speech_features import fbank

def progress(count, total, status=''):
        import sys
        bar_len = 60
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)

        sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
        sys.stdout.flush()


def normalize(data):
    #batch_inputs = sk.scale(batch_inputs, axis=1, with_mean=False, with_std=True)
    data_norm = data/(0.0001+np.transpose(np.tile(np.max(np.abs(data),axis=1),(np.size(data,1),1))))
    return data_norm


def augment(data):


    data_augment = normalize(data)
    # 0. add background sounds
    #data_augment = add_background_noise(data_augment)

    # 1. shift data
    data_augment = shift(data_augment, max_shift = 3000)

    # 2. stretch data
    data_augment = stretch(data_augment, rate_std = 0.1)

     # 3.add background audio to data

    # 4. Various kinds of audio fx
    data_augment = audio_fx(data_augment)

    #
    # 5. add noise to data
    data_augment = noise(data_augment, max_std = 0.05)

    return data_augment




#//////// Augmentation functions:

def audio_fx(data):

     m = np.size(data,0)
     n = np.size(data,1)

     data_fx = np.zeros((m,n))

     for i in range(0,m):
         pitch_shift = 0
         tempo_factor = 1
         od_gain = 0
         od_colour = 20;
         hp_freq = 30
         lp_freq = 7000
         wet_gain = -10;

         if np.random.rand()>0.7:
             pitch_shift = 400*np.random.rand()-200
         if np.random.rand()>0.7:
             tempo_factor = np.exp(6*np.random.rand()-3)
         if np.random.rand()>0.7:
             od_gain = 30*np.random.rand()
             od_colour = 40*np.random.rand()

         if np.random.rand()>0.7:
             lp_freq = 500+3000*np.random.rand() # range 500-3500 Hz
         if np.random.rand()>0.7:
             hp_freq = 80+320*np.random.rand() # range 30-400 Hz

         if np.random.rand()>0.5:
             wet_gain = 10-20*np.random.rand()  # range -10dB - 0dB
             apply_audio_effects = AudioEffectsChain()\
                .pitch(shift = pitch_shift)\
                .tempo(factor = tempo_factor)\
                .reverb(wet_gain=wet_gain)\
                .highpass(frequency=hp_freq)\
                .lowpass(frequency=lp_freq)\
                .overdrive(gain=od_gain, colour=od_colour)
         else:
             apply_audio_effects = AudioEffectsChain()\
                .pitch(shift = pitch_shift)\
                .tempo(factor = tempo_factor)\
                .highpass(frequency=hp_freq)\
                .lowpass(frequency=lp_freq)\
                .overdrive(gain=od_gain, colour=od_colour)

         temp = apply_audio_effects(np.squeeze(data[i,:]),channels_out = 1, sample_in = 16000, sample_out = 16000)[0:n]
         data_fx[i,0:len(temp)] = temp

     return data_fx

def stretch(data, rate_std):
    # data: input data
    # rate_std: standard deviation of stretches
    m = np.size(data,0)
    n = np.size(data,1)
    stretched_data = np.zeros((m,n))

    rates = np.random.normal(loc=1,scale=rate_std,size=m)

    for i in range(0,m):
        temp = librosa.effects.time_stretch(data[i,:], rates[i])
        if len(temp)>n:
            stretched_data[i,:] = temp[:n]
        else:
            stretched_data[i,:]  = np.pad(temp, (0, max(0, n - len(temp))), "constant")

    return stretched_data


def add_background_noise(data):
    background_noise_folder = getdirs()['noise']
    import glob, os
    from scipy.io import wavfile
    fs = 16000
    files = glob.glob(os.path.join(background_noise_folder, '*.wav'))
    noise_fragments = [wavfile.read(file)[1] for file in files]
    fragment_lengths = [len(frag) for frag in noise_fragments]
    noise_choices = np.random.choice(a=np.arange(len(noise_fragments)), size=len(data))
    noise_seek_choices = [np.minimum(int(np.random.rand()*fragment_lengths[choice]),fragment_lengths[choice]-fs) for choice in noise_choices]

    noise_matrix = np.concatenate([np.expand_dims(noise_fragments[noise_choices[i]][noise_seek_choices[i]:noise_seek_choices[i]+fs]
                                                  ,0)
                                   for i in range(len(data))])
    return data + noise_matrix


def getdirs():
    d = {}
    with open("folders.csv") as f:
        for line in f:
            (key, val) = line.strip().split(',')
            d[key] = val
    return d


def shift(data, max_shift):
    m = np.size(data,0)
    n = np.size(data,1)

    col_start = np.random.randint(0, max_shift, data.shape[0])
    idx = np.mod(col_start[:,None] + np.arange(n), n)

    shifted_data = np.zeros((m,n))
    shifted_data = data[np.arange(m)[:,None], idx]

    return shifted_data

def noise(data, max_std = 0.08):
    # ntype: noise type, pink is colored noise
    # max_std: maximum random standard deviation
    m = np.size(data,0)
    n = np.size(data,1)


    uneven = n%2
    X = np.random.randn(m,n//2+1+uneven) + 1j * np.random.randn(m,n//2+1+uneven)
    S = np.tile(np.sqrt(np.arange(np.size(X,-1))+1.),((np.size(data,0),1))) # +1 to avoid divide by zero
    noise_pink = (irfft(X/S)).real
    if uneven:
        noise_pink = noise_pink[:-1]

    noise_pink = noise_pink/np.std(noise_pink, axis=1, keepdims=True)
    noise_white = np.random.randn(m,n)

    # add pink noise:
    rand_amp = np.random.rand(m)*(np.random.rand(m)>0.5);
    factor = max_std*(np.transpose(np.tile(rand_amp*np.max(np.abs(data),axis=1),(np.size(data,1),1))))
    noised_data = data+factor*noise_pink;

    # add white noise:
    rand_amp = np.random.rand(m)*(np.random.rand(m)>0.5);
    factor = max_std*(np.transpose(np.tile(rand_amp*np.max(np.abs(data),axis=1),(np.size(data,1),1))))
    noised_data = noised_data+factor*noise_white;

    return noised_data

import scipy as sc

def augment_spectogram(data):
    # 1. stretch f (vocal tract warping)
    new_specs = []
    for i in range(len(data)):
        # alpha (stretch) between 0.8 and 1.2
        alpha= 0.8+0.4*np.random.rand()
        freq_warped = freq_warping(data[i],alpha=alpha) # stretch frequencies
        alpha = 0.8 + 0.4 * np.random.rand()
        time_warped = freq_warping(freq_warped.T,alpha=alpha).T # stretch times
        sigma = np.random.rand() * 0.5
        time_warped+= sigma*np.random.randn(np.size(data,1),np.size(data,2))
        new_specs.append(time_warped)

    new_specs = np.concatenate([np.expand_dims(spec,0) for spec in new_specs])

    # 3. add white noise

    return new_specs+sigma*np.random.randn(np.size(data,0),np.size(data,1),np.size(data,2))


def freq_warping(spec, alpha=1.0, f0=0.9, fmax=1):
    """ 'vocal tract warping': warp frequency axis
           Standard alpha between 0.9 and 1.1
    """

    freqbins = np.size(spec,0)
    timebins = np.size(spec,1)

    scale0 = np.linspace(0, 1, freqbins)

    # http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=650310&url=http%3A%2F%2Fieeexplore.ieee.org%2Fiel4%2F89%2F14168%2F00650310
    scale = np.array([x*alpha if x <= f0 else (fmax-alpha*f0)/(fmax-f0)*(x-f0)+alpha*f0 for x in scale0])
    scale = np.minimum(scale*(freqbins-1),freqbins-1)

    Xq = np.repeat(np.expand_dims(scale,axis=1),timebins,axis=1)
    Yq = np.repeat(np.expand_dims(np.arange(0,timebins),axis=0),freqbins,axis=0)

    spline_order = 3
    newspec = sc.ndimage.map_coordinates(spec, [Xq.ravel(), Yq.ravel()], order=spline_order, mode='nearest').reshape((freqbins,timebins))

    return newspec

class threadsafe_iter(object):
    """
    Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """
    A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


def convert_to_spectrogram(data, a=1e-3):
    '''
    performs spectrogram analysis
    :param data
    :return: spectrogram
    '''

    out = []
    for d in data:
        features, energy  = fbank(d,16000, winlen=0.05, winstep=0.004, nfft=2048, nfilt=256,winfunc=np.hamming, preemph=0.5)
        fbank_feat= np.log(features+a)-np.log(a)
        out.append(np.expand_dims(fbank_feat, 0))
    return np.concatenate(out)
