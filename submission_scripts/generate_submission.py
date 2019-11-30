import numpy as np
import pandas as pd
from data_generator import GeneratorFactory
from keras.models import load_model
from utils  import normalize, augment
import os
from utils import getdirs
data_directory = '../data'
batch_size = 256
factory = GeneratorFactory(other_class_rate=0.09, silence_class_rate=0.09, spectrogram='no',data_dir = getdirs()['data'])
m = load_model('../logs_can_20171221_102257/model_e25_loss_0.38089083498308607_acc0.8834651077519007')
lookup = np.array(['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go','silence','unknown'])

# targets = ['bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go',
#        'happy', 'house', 'left', 'marvin', 'nine', 'no', 'off', 'on',
#        'one', 'right', 'seven', 'sheila', 'six', 'stop',
#        'three', 'tree', 'two', 'up', 'wow', 'yes', 'zero','silence','other']
targets = np.array(['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go','silence','unknown'])
fnames = []
preds = []
probs = []
cnt = 0
for X, file_names in factory.holdout_generator(batch_size):
    if len(X) == 0:
        continue

    y = m.predict(X)
    probs.append(y)

    for i in range(len(file_names)):
        fnames.append(file_names[i][file_names[i].find(os.sep)+1:])
        pred = targets[np.argmax(y[i])] if targets[np.argmax(y[i])] in lookup else 'unknown'
        preds.append(pred)

    cnt += 1
    print(cnt)



preds = np.concatenate([np.expand_dims(p, 0) for p in preds])

data = {'fname':fnames,'label':preds}
df = pd.DataFrame(data=data)
df.set_index('fname', inplace=True)
df.to_csv('submission.csv')
probs = pd.DataFrame(data=np.concatenate(probs),columns=targets, index = fnames)
pd.concat([df,probs],axis=1).to_csv('submission_with_probabilities.csv')