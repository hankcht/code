import matplotlib.pyplot as plt
import numpy as np
import os
import keras
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
import random
import datetime
import pandas as pd
from glob import glob
import matplotlib
import argparse
from A1_TrainAndRunCNN import output_cut_value # We import the network output cut variable from A1 
from NuRadioReco.utilities import units

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

total_station = np.load(f'/pub/tangch3/ARIANNA/DeepLearning/circled_data/Station_Traces/{amp_type}/Stn{station_id}_Traces.npy')
total_SNR = np.load(f'/pub/tangch3/ARIANNA/DeepLearning/circled_data/Station_SNR/{amp_type}/Stn{station_id}_SNR.npy') # Obtained from Chi-SNRgraph.py (11/29/2024)
total_Chi = np.load(f'/pub/tangch3/ARIANNA/DeepLearning/circled_data/Station_Chi/{amp_type}/Stn{station_id}_Chi.npy')

print(f'total circled events: {len(total_station)}')

selected_chi = [] 
selected_chi_idx = []
for i, chi in enumerate(total_Chi):
    if chi >= 0.5:                  # our selection criterion here is events with Chi >= 0.5 (6/20/2024)
        selected_chi.append(chi)
        selected_chi_idx.append(i)

selected_station = np.array(total_station[selected_chi_idx]) 

print(f'selected station: {selected_station.shape}')

# To get passed events, we must first run our trained CNN on all the circled events
model = keras.models.load_model(f'/pub/tangch3/ARIANNA/DeepLearning/models/200s_time/data_data_2024-10-24_15-38-14_RCR_Backlobe_model_2Layer.h5') # copy from /models
simordata = 'data' # depending on the model, change the name of the N.O. histogram

prob_station = model.predict(selected_station) # change the input if needed

print(f'output cut value: {output_cut_value}')

# =================================================================================================================================
# Here we get useful information about how our model did with out network output cut
# note: depending on the input into the model, our interpretation below is different. We could just have passed events for example.
# (11/29/2024) We used selected_station, so the passed events are actually selected-passed events
# =================================================================================================================================

number_passed_events = (prob_station > output_cut_value).sum() # number of events passing output cut
passed_events_efficiency = number_passed_events / len(prob_station)
passed_events_efficiency = passed_events_efficiency.round(decimals=4)*100 # efficiency

print(f'number of passed events: {number_passed_events}')
print(f'Output cut efficiency: {passed_events_efficiency}')

plot_folder = '/pub/tangch3/ARIANNA/DeepLearning/plots/Candidates'

fig, ax = plt.subplots(figsize=(8, 6))
hist_values, bin_edges, _ = ax.hist(prob_station, bins=20, range=(0, 1), histtype='step', color='green', linestyle='solid', label='output', density=False)
ax.set_xlabel('Network Output', fontsize=18)
ax.set_ylabel('Number of Events', fontsize=18)
# ax.set_yscale('log') # Set logarithmic scale for y-axis, if needed
# ax.set_ylim(1, max(10 ** (np.ceil(np.log10(hist_values))))) # uncomment if needed for semi-log plot
ax.set_title(f'Station {station_id} network output (200s_time)')
ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.tick_params(axis='both', which='both', labelsize=12, length=5)
# ax.tick_params(axis='y', which='major', labelsize=12, length=2)
ax.text(0.05, 0.93, f'Total Passed Events: {number_passed_events}', transform=ax.transAxes, fontsize=12)
ax.text(0.05, 0.87, f'Total Selected Events: {len(selected_station)}', transform=ax.transAxes, fontsize=12)
ax.text(0.05, -0.12, 'BL', verticalalignment='center', horizontalalignment='center', fontsize=12, transform=ax.transAxes, color='blue')
ax.text(0.96, -0.12, 'RCR', verticalalignment='center', horizontalalignment='center', fontsize=12, transform=ax.transAxes, color='red')
plt.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)
ax.axvline(x=output_cut_value, color='y', label='cut')
ax.legend(loc='upper right', fontsize=8, bbox_to_anchor=(0.85, 0.99))
plt.savefig(f'{plot_folder}/Station {station_id}/{simordata}_timeStn{station_id}histogram.png')
print(f'saving to {plot_folder}')
print('Done!')

















# Unused stuff
selected_events_dict = {
'stn14selectedevents' : [3, 6, 13, 15, 18, 19, 21, 22, 24, 28, 35, 39, 40, 49],
'stn17selectedevents' : [1, 3, 18, 21, 22, 26, 28, 29, 31, 33, 44, 45, 47, 48, 49, 50],
'stn19selectedevents' : [1, 2, 4, 5, 7, 15, 17, 18, 19, 25, 27, 31, 34], 
'stn30selectedevents' : [4, 6, 7, 8, 9, 11, 12, 13, 16, 17, 20, 22, 25, 26, 29, 30, 32, 33, 34, 36, 38, 43, 49, 50, 55, 60, 61, 62]
}
