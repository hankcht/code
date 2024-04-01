import matplotlib.pyplot as plt
import numpy as np
import os
import keras
import time
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D, Activation, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
#from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
import random
import datetime
import pandas as pd
from glob import glob
import matplotlib

#Set parameters
path = f'/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/'    #Edit path to properly point to folder
model_path = f'/data/homezvol3/tangch3/ARIANNA/DeepLearning/models/'  #Path to save models
plots_path_accuracy = '/data/homezvol3/tangch3/ARIANNA/DeepLearning/plots/accuracy/'
plots_path_loss = '/data/homezvol3/tangch3/ARIANNA/DeepLearning/plots/loss/'  #Path to save plots
amp = '200s'                                                                  #Set which amplifier to run on
RCR_path = f'simulatedRCRs/{amp}_2.9.24/'
backlobe_path = f'simulatedBacklobes/{amp}_2.9.24/'
TrainCut = 5000                                                               #Number of events to use for training

def load_data():
    RCR_files = []
    print(f'path {path + RCR_path}')
    for filename in os.listdir(path + RCR_path):
        print(f'filename {filename}')
        if filename.startswith(f'FilteredSimRCR_{amp}_'):
            print(f'appending')
            RCR_files.append(path + RCR_path +  filename)
    RCR = np.empty((0, 4, 256))
    print(f'rcr files {RCR_files}')
    for file in RCR_files:
        print(f'RCR file {file}')
        RCR_data = np.load(file)[0:, 0:4]
        print(f'RCR data shape {RCR_data.shape} and RCR shape {RCR.shape}')
        RCR = np.concatenate((RCR, RCR_data))
    
    Backlobes_files = []
    for filename in os.listdir(path + backlobe_path):
        if filename.startswith(f'Backlobe_{amp}_'):
            Backlobes_files.append(path + backlobe_path + filename)
    Backlobe = np.empty((0, 4, 256))
    for file in Backlobes_files:
        print(f'Backlobe file {file}')
        Backlobe_data = np.load(file)[0:, 0:4]
        Backlobe = np.concatenate((Backlobe, Backlobe_data))
    
    return (RCR, Backlobe) 

RCR, Backlobe = load_data()
RCR, Backlobe = RCR[5000:], Backlobe[5000:]

# Load the model, copy from /models
model = keras.models.load_model(f'/data/homezvol3/tangch3/ARIANNA/DeepLearning/models/2024-03-07_22-58-07_RCR_Backlobe_model_2Layer_1.h5')

prob_RCR = model.predict(RCR)
#to get one for RCR
prob_RCR = 1 - prob_RCR

prob_Backlobe = model.predict(Backlobe)
#to get zero for Backlobe
prob_Backlobe = 1 - prob_Backlobe

# Generate some random data between 0 and 1
# data1 = np.random.rand(10000)
# data2 = np.random.rand(10000)

# Create histogram
# plt.hist(data, bins=50, range=(0, 1))

dense_val = False
hist_values, bin_edges, _ = plt.hist(prob_Backlobe, bins=20, range=(0, 1), histtype='step', color='red', linestyle='solid', label='Backlobe', density=dense_val)
plt.hist(prob_RCR, bins=20, range=(0, 1), histtype='step',color='blue', linestyle='solid',label='RCR',density = dense_val)

# Set logarithmic scale for y-axis
plt.yscale('log')

# Set labels and title
plt.xlabel('Network Output', fontsize = 18)
plt.ylabel('Number of Events', fontsize = 18)
plt.title('RCR vs Backlobe network output')
plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0],size=18)
plt.yticks(size=18)
plt.ylim(1, max(10 ** (np.ceil(np.log10(hist_values)))))
plt.tick_params(axis='both', which='major', labelsize=13)
plt.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)
# handles, labels = plt.get_legend_handles_labels()
# new_handles = [Line2D([], [], c=h.get_edgecolor(), linestyle=h.get_linestyle()) for h in handles]
plt.legend(loc='upper center', fontsize=16)
# Save the plot to a file (in PNG format)
plt.savefig('DeepLearning/plots/network_output/histogram.png')

#include RCR efficiency RCR files/total files
#now train on frequency
#parameters to change: window size
#might want to have less than window size 10 for frequency (maybe 8) window size 10 means 10 samples, so over 5ns
