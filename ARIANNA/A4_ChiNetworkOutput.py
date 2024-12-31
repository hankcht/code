import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow
import keras
import time
from tensorflow import keras
from keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D, Activation, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
#from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
import random
import datetime
import pandas as pd
from glob import glob
import argparse
from pathlib import Path
from NuRadioReco.utilities import units
from NuRadioReco.utilities.io_utilities import read_pickle
from matplotlib.markers import MarkerStyle
import templateCrossCorr as txc
from A0_Utilities import load_sim, getMaxChi

#this file creates Chi-NetworkOutput plots of:
#   [1] simualted RCR(blue) & backlobe(red) 
#   [2] station data that passed Ryan's cuts
#   [3] selected station data
#   [4] all station data 

#total events := events that passed Ryan's cuts
#selected events := events that we picked (Chi > 0.5)
#passed events := events that passes the network output cut (>0.95)  

#Set parameters
path = f'/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/'    #Edit path to properly point to folder
model_path = f'/pub/tangch3/ARIANNA/DeepLearning/models/'  #Path to save models
amp = '200s'                                                                  #Set which amplifier to run on
RCR_path = f'simulatedRCRs/{amp}_2.9.24/'
backlobe_path = f'simulatedBacklobes/{amp}_2.9.24/'
output_cut_value = 0.95 #Change this depending on chosen cut


def loadTemplate(type='RCR', amp='200s'):
    if type == 'RCR':
        if amp == '200s':
                templates_RCR = '../../../../../dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/templates/reflectedCR_template_200series.pkl'
                templates_RCR = read_pickle(templates_RCR)
                for key in templates_RCR:
                    temp = templates_RCR[key]
                templates_RCR = temp
                return templates_RCR
        elif amp == '100s':
                templates_RCR = '../../../../../dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/templates/reflectedCR_template_100series.pkl'
                templates_RCR = read_pickle(templates_RCR)
                for key in templates_RCR:
                    temp = templates_RCR[key]
                templates_RCR = temp
                return templates_RCR

    print(f'{type} {amp} not implemented')
    quit()

#Section 1: simulation

RCR, Backlobe = load_sim()
RCR, Backlobe = RCR[5000:], Backlobe[5000:] #I used the first 5000 simulation for training, so I run the trained model on the remaining

# Load the model, copy from /models
model = keras.models.load_model(f'/pub/tangch3/ARIANNA/DeepLearning/models/200s_time/data_data_2024-10-12_20-05-36_RCR_Backlobe_model_2Layer.h5')

prob_RCR = model.predict(RCR)

prob_Backlobe = model.predict(Backlobe)

print(f'number of netoutput events is RCR: {len(prob_RCR)} and Backlobe: {len(prob_Backlobe)}')

# Now that we have the Network Output of sim RCR and Backlobe, we need their Chi values

def plotSimChiNetworkOutput(type='RCR'):

    templates_RCR = loadTemplate(type='RCR', amp=amp)

    path = '../../../../../dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/'
    simulation_date = '2.9.24'

    RCR_files = []
    if type == 'RCR':
        path += f'simulatedRCRs/{amp}_{simulation_date}/'
        for filename in os.listdir(path):
            if filename.startswith(f'FilteredSimRCR'):
                RCR_files.append(os.path.join(path, filename))
    elif type == 'Backlobe':
        path += f'simulatedBacklobes/{amp}_{simulation_date}/'
        for filename in os.listdir(path):
            if filename.startswith(f'Backlobe'):
                RCR_files.append(os.path.join(path, filename))

    for file in RCR_files:
        RCR_sim = np.load(file)  

    sim_Chi = []

    for iR, RCR in enumerate(RCR_sim):
        traces = []
        for trace in RCR:
            traces.append(trace * units.V)
        sim_Chi.append(getMaxChi(traces, 2*units.GHz, templates_RCR , 2*units.GHz))


    print(f'len sim chi {len(sim_Chi)}')

    sim_Chi = np.array(sim_Chi)

    sim_Chi = sim_Chi[5000:]

    folder = f'/pub/tangch3/ARIANNA/DeepLearning/plots/Chi_NetworkOutput/Simulation' 
    
    if type == 'RCR':
        plt.scatter(sim_Chi, prob_RCR, label=f'{len(prob_RCR)} RCR Simulations', facecolor='blue', edgecolor='none')
        plt.xlim((-0.05, 1.05))
        plt.ylim((-0.05, 1.05))
        plt.xlabel('Chi')
        plt.ylabel('Network Output')
        plt.legend()    
        plt.tick_params(axis='x', which='minor', bottom=True)
        plt.grid(visible=True, which='both', axis='both')
        plt.title(f'Simulation Chi-NetworkOutput')
        plt.text(0, 1.1, 'RCR', 
            verticalalignment='center', 
            horizontalalignment='center',
            fontsize=12, 
            color='black')
        plt.text(0, -0.15, 'BL', 
            verticalalignment='center', 
            horizontalalignment='center',
            fontsize=12, 
            color='black')
        print(f'Saving {folder}/sim_RCR.png')
        plt.axhline(y = output_cut_value, color = 'y', label = 'cut')
        plt.savefig(f'{folder}/data_RCR.png')
        plt.clf()
        plt.close()
    elif type == 'Backlobe':
        plt.scatter(sim_Chi, prob_Backlobe, label=f'{len(prob_Backlobe)} BL Simulations', facecolor='red', edgecolor='none')
        plt.xlim((-0.05, 1.05))
        plt.ylim((-0.05, 1.05))
        plt.xlabel('Chi')
        plt.ylabel('Network Output')
        plt.legend()    
        plt.tick_params(axis='x', which='minor', bottom=True)
        plt.grid(visible=True, which='both', axis='both')
        plt.title(f'Simulation Chi-NetworkOutput')
        plt.text(0, 1.1, 'RCR', 
            verticalalignment='center', 
            horizontalalignment='center',
            fontsize=12, 
            color='black')
        plt.text(0, -0.15, 'BL', 
            verticalalignment='center', 
            horizontalalignment='center',
            fontsize=12, 
            color='black')
        print(f'Saving {folder}/sim_BL.png')
        plt.axhline(y = output_cut_value, color = 'y', label = 'cut')
        plt.savefig(f'{folder}/data_BL.png')
        # plt.clf()
        # plt.close()

noiseRMS = 22.53 * units.mV
plotSimChiNetworkOutput(type='Backlobe')
plotSimChiNetworkOutput(type='RCR')


# Section 2: Events passing noise cuts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Analysis on particular station')
    parser.add_argument('station', type=int, default=19, help='Station to run on')
    args = parser.parse_args()
    station_id = args.station

    if station_id in [14, 17, 19, 30]:
        amp_type = '200s'
    elif station_id in [13, 15, 18]:
        amp_type = '100s'

    #We can first find Chi
    data = np.load(f'../../../../../dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/station_data/Station{station_id}_SNR_Chi.npy', allow_pickle=True)
    plot_folder = f'/pub/tangch3/ARIANNA/DeepLearning/plots/Chi_NetworkOutput/Station_{station_id}' 
    Path(plot_folder).mkdir(parents=True, exist_ok=True)

    All_SNRs = data[0]
    All_RCR_Chi = data[1]
    All_Azi = data[2]
    All_Zen = data[3]
    PassingCut_SNRs = data[4]
    PassingCut_RCR_Chi = data[5]
    PassingCut_Azi = data[6]
    PassingCut_Zen = data[7]

    print(f'total events: {len(data[5])} & {len(data[1])}')

    #Now I find Network Output
    station_path = f'/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/eventsPassingNoiseCuts/'

    #Load station data
    def load_data():
        station_files = []
        print(f'path {station_path}')
        for filename in os.listdir(station_path):
            print(f'filename {filename}')
            if filename.startswith(f'Station{station_id}'):
                print(f'appending')
                station_files.append(station_path +  filename)
        station = np.empty((0, 4, 256))
        print(f'station files {station_files}')
        for file in station_files:
            print(f'station file {file}')
            station_data = np.load(file)[0:, 0:4]
            print(f'station data shape {station_data.shape} and station shape {station.shape}')
            station = np.concatenate((station, station_data))

        return (station, station_data.shape) 

    station, total_number_events = load_data()
    print(f'total events: {len(station)}')

    # Load the model, copy from /models
    model = keras.models.load_model(f'/pub/tangch3/ARIANNA/DeepLearning/models/200s_time/data_data_2024-10-12_20-05-36_RCR_Backlobe_model_2Layer.h5')

    prob_station = model.predict(station) 

    plt.scatter(PassingCut_RCR_Chi, prob_station, label=f'{len(prob_station)} Data Events', facecolor='black', edgecolor='none')
    plt.xlim((-0.05, 1.05))
    plt.ylim((-0.05, 1.05))
    plt.xlabel('Chi')
    plt.ylabel('Network Output')
    plt.legend()    
    plt.tick_params(axis='x', which='minor', bottom=True)
    plt.grid(visible=True, which='both', axis='both')
    plt.title(f'(After-Cut) Data Chi-NetworkOutput')
    plt.text(0, 1.1, 'RCR', 
        verticalalignment='center', 
        horizontalalignment='center',
        fontsize=12, 
        color='black')
    plt.text(0, -0.15, 'BL', 
        verticalalignment='center', 
        horizontalalignment='center',
        fontsize=12, 
        color='black')
    print(f'Saving {plot_folder}/Stn{station_id}.png')
    plt.axhline(y = output_cut_value, color = 'y', label = 'cut')
    plt.savefig(f'{plot_folder}/data_Stn{station_id}.png')
    plt.clf()
    plt.close()    

    #Section 3: If I want selected events with Chi > 0.5 (could do the same thing for Chi with PassingCut_Chis),
    selected_positions = []

    for index, value in enumerate(PassingCut_RCR_Chi):
        if value >= 0.5:
            selected_positions.append(index)
    
    print(f'the index of selected data events are: {selected_positions}')

    selected_station = station[selected_positions]

    prob_selected_station = model.predict(selected_station)

    #We can find Chi values of these selected events
    selected_Chi = [PassingCut_RCR_Chi[index] for index in selected_positions]

    plt.scatter(selected_Chi, prob_selected_station, label=f'{len(prob_selected_station)} Data Events', facecolor='black', edgecolor='none')
    plt.xlim((-0.05, 1.05))
    plt.ylim((-0.05, 1.05))
    plt.xlabel('Chi')
    plt.ylabel('Network Output')
    plt.legend()    
    plt.tick_params(axis='x', which='minor', bottom=True)
    plt.grid(visible=True, which='both', axis='both')
    plt.title(f'Selected Data Chi-NetworkOutput')
    plt.text(0, 1.1, 'RCR', 
        verticalalignment='center', 
        horizontalalignment='center',
        fontsize=12, 
        color='black')
    plt.text(0, -0.15, 'BL', 
        verticalalignment='center', 
        horizontalalignment='center',
        fontsize=12, 
        color='black')
    print(f'Saving {plot_folder}/selected_Stn{station_id}.png')
    plt.axhline(y = output_cut_value, color = 'y', label = 'cut')
    plt.savefig(f'{plot_folder}/data_selected_Stn{station_id}.png')
    plt.clf()
    plt.close()    

# Section 4: All station data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Analysis on particular station')
    parser.add_argument('station', type=int, default=14, help='Station to run on')
    args = parser.parse_args()
    station_id = args.station

    if station_id in [14, 17, 19, 30]:
        amp_type = '200s'
        noiseRMS = 22.53 * units.mV
    elif station_id in [13, 15, 18]:
        amp_type = '100s'
        noiseRMS = 20 * units.mV

    #load data
    data_path = f'../../../../../dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/station_data/events_prepared_for_model_training/'

    all_data_files = []

    for filename in os.listdir(data_path):
        if filename.startswith(f'FilteredStation{station_id}') and filename.endswith(('part0.npy','part1.npy')):
            print(f'appending {filename}')
            all_data_files.append(os.path.join(data_path, filename))

    print(f'size of data: {len(all_data_files)}') 
    # print(f'{all_data_files}')

    shapes = [np.load(file).shape for file in all_data_files]
    total_rows = sum(shape[0] for shape in shapes)
    first_shape = shapes[0][1:]
    print(f'first shape is: {first_shape}')
    all_station_data = np.empty((total_rows, *first_shape), dtype=np.float32)

    start_idx = 0

    for i, file in enumerate(all_data_files):
        data = np.load(file)
        num_rows = data.shape[0]

        all_station_data[start_idx:start_idx + num_rows] = data
   
        start_idx += num_rows   

    print(f'total # of data: {len(all_station_data)}')
    print(f'size of data: {all_station_data.shape}')

    empty = 0
    empty_mask = np.zeros(len(all_station_data), dtype=bool)
    for iD, d in enumerate(all_station_data):
        if isinstance(d, np.ndarray):
            empty_mask[iD] = np.max(d) != 0
            if not empty_mask[iD]:
                empty += 1
        else:
            raise TypeError(f"Expected d to be a NumPy array, got {type(d)}")
    print(f'empty {empty} & non-empty {len(all_station_data) - empty}')

    masked_all_station_data = all_station_data[empty_mask] # Reduces data down to only events that have non-zero content

    # print(f'mask size: {empty_mask.shape}')
    # print(f'size of masked data: {masked_all_station_data.shape}')
    print(tensorflow.__version__)

    templates_RCR = loadTemplate(type='RCR', amp=amp)

    all_data_Chi = []

    for iR, data in enumerate(masked_all_station_data):
        traces = []
        for trace in data:
            traces.append(trace * units.V)
        all_data_Chi.append(getMaxChi(traces, 2*units.GHz, templates_RCR, 2*units.GHz))

    print(f'total # of data: {len(all_data_Chi)}')

    # Load the model, copy from /models
    model = keras.models.load_model(f'/pub/tangch3/ARIANNA/DeepLearning/models/200s_time/data_data_2024-10-12_20-05-36_RCR_Backlobe_model_2Layer.h5')

    prob_all_station = model.predict(masked_all_station_data) 

    all_plot_folder = f'/pub/tangch3/ARIANNA/DeepLearning/plots/Chi_NetworkOutput/Station_{station_id}'

    plt.scatter(all_data_Chi, prob_all_station, label=f'{len(prob_all_station)} Data Events', facecolor='black', edgecolor='none')
    plt.ylim((-0.05, 1.05))
    plt.xlabel('Chi')
    plt.ylabel('Network Output')
    plt.legend()    
    plt.tick_params(axis='x', which='minor', bottom=True)
    plt.grid(visible=True, which='both', axis='both')
    plt.title(f'All Station {station_id} Data Chi-NetworkOutput')
    plt.text(0, 1.1, 'RCR', 
       verticalalignment='center', 
       horizontalalignment='center',
       fontsize=12, 
       color='black')
    plt.text(0, -0.15, 'BL', 
       verticalalignment='center', 
       horizontalalignment='center',
       fontsize=12, 
       color='black')
    plt.axhline(y = output_cut_value, color = 'y', label = 'cut')
    print(f'Saving {all_plot_folder}/All_Stn{station_id}.png')
    plt.savefig(f'{all_plot_folder}/data_All_Stn{station_id}.png')
    print('Done!')
    plt.clf()
    plt.close()


########################################################################

    #load data
    data_path = f'../../../../../dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/station_data/events_prepared_for_model_training/'

    all_data_files = []

    for filename in os.listdir(data_path):
        if filename.startswith(f'FilteredStation{station_id}') and filename.endswith(('part2.npy','part3.npy','part4.npy')):
            print(f'appending {filename}')
            all_data_files.append(os.path.join(data_path, filename))

    print(f'size of data: {len(all_data_files)}') 
    # print(f'{all_data_files}')

    shapes = [np.load(file).shape for file in all_data_files]
    total_rows = sum(shape[0] for shape in shapes)
    first_shape = shapes[0][1:]
    print(f'first shape is: {first_shape}')
    all_station_data = np.empty((total_rows, *first_shape), dtype=np.float32)

    start_idx = 0

    for i, file in enumerate(all_data_files):
        data = np.load(file)
        num_rows = data.shape[0]

        all_station_data[start_idx:start_idx + num_rows] = data
   
        start_idx += num_rows   

    print(f'total # of data: {len(all_station_data)}')
    print(f'size of data: {all_station_data.shape}')

    empty = 0
    empty_mask = np.zeros(len(all_station_data), dtype=bool)
    for iD, d in enumerate(all_station_data):
        if isinstance(d, np.ndarray):
            empty_mask[iD] = np.max(d) != 0
            if not empty_mask[iD]:
                empty += 1
        else:
            raise TypeError(f"Expected d to be a NumPy array, got {type(d)}")
    print(f'empty {empty} & non-empty {len(all_station_data) - empty}')

    masked_all_station_data = all_station_data[empty_mask] # Reduces data down to only events that have non-zero content

    # print(f'mask size: {empty_mask.shape}')
    # print(f'size of masked data: {masked_all_station_data.shape}')
    print(tensorflow.__version__)

    all_data_Chi = []
    templates_RCR = loadTemplate(type='RCR', amp=amp)

    for iR, data in enumerate(masked_all_station_data):
        traces = []
        for trace in data:
            traces.append(trace * units.V)
        all_data_Chi.append(getMaxChi(traces, 2*units.GHz, templates_RCR, 2*units.GHz))

    print(f'total # of data: {len(all_data_Chi)}')

    # Load the model, copy from /models
    model = keras.models.load_model(f'/pub/tangch3/ARIANNA/DeepLearning/models/200s_time/2024-04-11_15-56-07_RCR_Backlobe_model_2Layer_0.h5')

    prob_all_station2 = model.predict(masked_all_station_data) 

    all_plot_folder = f'/pub/tangch3/ARIANNA/DeepLearning/plots/Chi_NetworkOutput/Station_{station_id}'

    plt.scatter(all_data_Chi, prob_all_station2, label=f'{657698 + len(prob_all_station2)} Data Events', facecolor='black', edgecolor='none')
    plt.xlim((-0.05, 1.05))
    plt.ylim((-0.05, 1.05))
    plt.xlabel('Chi')
    plt.ylabel('Network Output')
    plt.legend()    
    plt.tick_params(axis='x', which='minor', bottom=True)
    plt.grid(visible=True, which='both', axis='both')
    plt.title(f'All Station {station_id} Data Chi-NetworkOutput')
    plt.text(0, 1.1, 'RCR', 
        verticalalignment='center', 
        horizontalalignment='center',
        fontsize=12, 
        color='black')
    plt.text(0, -0.15, 'BL', 
        verticalalignment='center', 
        horizontalalignment='center',
        fontsize=12, 
        color='black')
    plt.axhline(y = output_cut_value, color = 'y', label = 'cut')
    print(f'Saving {all_plot_folder}/All_Stn{station_id}.png')
    plt.savefig(f'{all_plot_folder}/All_Stn{station_id}.png')
    print('Done!')
    plt.clf()
    plt.close()




























    
    
