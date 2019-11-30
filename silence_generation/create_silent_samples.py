from scipy.io import wavfile
import glob, os, random, string
import os
noise_dir = 'D:\Mustafa\workspace\data\\train\_background_noise_'
silence_dir = 'D:\Mustafa\workspace\data\\train\\audio\silence'
files = glob.glob(os.path.join(noise_dir,'*.wav'))


# go over each file, extracting 1 second segments, with 80% overlap.
# Assign the first 70% to training, the second 20% for testing and last 10% for validation.
# Keeping chronological order to prevent data bleed


for file in files:

    input = wavfile.read(file)
    fs = input[0]
    data = input[1]
    n_overlap = int(0.2*fs)
    for i in range(0,len(data),n_overlap):
        if i < 0.7*len(data):
            prefix = 'train_'
        elif i < 0.9*len(data):
            prefix = 'test_'
        else:
            prefix = 'val_'

        file_name = prefix + ''.join(random.choices(string.ascii_uppercase + string.digits, k=10)) + '.wav'
        wavfile.write(os.path.join(silence_dir, file_name), rate=fs, data=data[i:i+fs])


