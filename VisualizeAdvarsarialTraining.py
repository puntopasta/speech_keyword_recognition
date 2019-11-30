import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
logdir = 'logs_can_20171218_103038'

def plot_adversarial_progress(log):
    
    f, (ax1,ax2) = plt.subplots(1, 2, figsize=(12, 4))
    f.subplots_adjust(wspace=0.5)
    E_A_connect = [val for pair in zip(log.E_A_loss.values, log.E_A_loss_Eupdate.values) for val in pair]

    ax1.plot(np.arange(0, len(log.E_C_loss.values)), log.E_C_loss.values, '-',
            color=(0,0,0), antialiased=True, label='Classification loss')
    
    ax1b = ax1.twinx(); ax1b.set_ylim([0,1]);  
    ax1b.plot(np.arange(0, len(log.train_C_acc.values)), log.train_C_acc.values, '--',
            color=(72 / 255, 152 / 255, 211 / 255), antialiased=True, label='Classification train accuracy')
    
    ax1c = ax1.twinx(); ax1c.set_ylim([0,1]);  
    ax1c.plot(np.arange(0, len(log.train_A_acc.values)), log.train_A_acc.values, '-.',
            color=(164 / 255, 63 / 255, 152 / 255), antialiased=True, label='Adversarial train accuracy')

    ax1.set_xlabel('Iteration #')
    ax1.set_ylabel('Loss', size=14)
    ax1b.set_ylabel('Training accuracy', size=14)
    
    handles, labels = ax1.get_legend_handles_labels()
    handles1b, labels1b = ax1b.get_legend_handles_labels()
    handles1c, labels1c = ax1c.get_legend_handles_labels()

    lg = ax1.legend(handles+handles1b+handles1c, labels+labels1b+labels1c)
    lg.draw_frame(False)

    E_A_connect = [val for pair in zip(log.E_A_loss.values, log.E_A_loss_Eupdate.values) for val in pair]

    ax2.plot(.5 * np.arange(0, len(E_A_connect)), E_A_connect, '-', color=(0.8, 0.8, 0.8), antialiased=True)
    ax2.plot(1 * np.arange(0, len(log.E_A_loss.values)), log.E_A_loss.values, 's',
            color=(72 / 255, 152 / 255, 211 / 255), antialiased=True, label='Adversarial update')
    ax2.plot(1 * np.arange(0, len(log.E_A_loss_Eupdate.values)) + .5, log.E_A_loss_Eupdate.values, 'o',
            color=(164 / 255, 63 / 255, 152 / 255), antialiased=True, label='Encoder update')

    ax2.set_xlabel('Iteration #')
    ax2.set_ylabel('Loss', size=14)
    handles, labels = ax2.get_legend_handles_labels()
    lg = ax2.legend(handles, labels)
    lg.draw_frame(False)

    plt.show(block=True)
    plt.pause(0.05)


plot_adversarial_progress((pd.read_csv(os.path.join(logdir, 'training_log.csv')).set_index(['epoch','batch'])))
print((pd.read_csv(os.path.join(logdir, 'test_log.csv')).set_index(['epoch','batch'])))||||||| .r43
