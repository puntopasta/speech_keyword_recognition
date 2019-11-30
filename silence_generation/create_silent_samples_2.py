from scipy.io import wavfile
import glob, os, string
import numpy as np
noise_dir = 'D:\Mustafa\workspace\data\\train\_background_noise_'
silence_dir = 'D:\Mustafa\workspace\data\\train\\audio\silence'
files = glob.glob(os.path.join(noise_dir,'*.wav'))

fs = 16000
for i in np.arange(16000): # generate 16k samples to match the training set
    sample = np.array(np.zeros(fs), dtype='int16')
    prefix = np.random.choice(a=['train_','test_','validation_'],size=1,p=[0.7,0.2,0.1])[0]
    file_name = prefix+''.join(np.random.choice(np.array(np.array([s for s in string.ascii_uppercase]), dtype='str'), size=10).tolist()) + '.wav'
    for file in files:
        if np.random.rand() > 0.35:
            continue

        sig = wavfile.read(file)[1]
        seek = int(np.random.rand() * (len(sig) - fs))
        sample += sig[seek:seek+fs]

    if np.random.rand() > 0.8:
        sample *= np.array(np.random.rand()*2,dtype='int16')

    wavfile.write(os.path.join(silence_dir, file_name), rate=fs, data=sample)


