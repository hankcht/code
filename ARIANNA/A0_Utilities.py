import os
import time
import glob
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, Flatten, Dense
import matplotlib.pyplot as plt
import pickle
import templateCrossCorr as txc
import NuRadioReco
from NuRadioReco.utilities import units, fft

    
def getMaxChi(traces, sampling_rate, template_trace, template_sampling_rate, parallelChannels=[[0, 2], [1, 3]]):
    # note: variable parallelChannels uses a list as default, which may lead to unexpected results. Avoid this
    
    # Parallel channels should be index corresponding to the channel in traces
    maxCorr = []
    for parChans in parallelChannels:
        parCorr = 0
        for chan in parChans:
            xCorr = txc.get_xcorr_for_channel(traces[chan], template_trace, sampling_rate, template_sampling_rate)
            parCorr += np.abs(xCorr)
        maxCorr.append(parCorr / len(parChans))

    return max(maxCorr)

def getMaxSNR(traces, noiseRMS=22.53 * units.mV):

    SNRs = []
    for trace in traces:
        p2p = (np.max(trace) + np.abs(np.min(trace))) * units.V
        SNRs.append(p2p / (2*noiseRMS))
    
    if max(SNRs)==0:
        print(f'zero error')
        SNRs = []
        for trace in traces:
            print(f'trace {trace}')
            p2p = (np.max(trace) + np.abs(np.min(trace))) * units.V
            SNRs.append(p2p / (2*noiseRMS))

    # quit()
    return max(SNRs)

def load_data(type, amp_type, station_id):

    data_folder = f'/pub/tangch3/ARIANNA/DeepLearning/{type}'

    if type == 'All_data':
        print(f'using {type} files')
        All_data_SNR = np.load(f'{data_folder}/Station_SNR/{amp_type}/Stn{station_id}_SNR.npy')
        All_data_Chi = np.load(f'{data_folder}/Station_Chi/{amp_type}/Stn{station_id}_Chi.npy')

        temporary_Traces = []
        for filename in os.listdir(f'{data_folder}/Station_Traces/{amp_type}/'):
            if filename.startswith(f'Stn{station_id}'):
                print(filename)
                temporary_Traces.append(np.load(f'{data_folder}/Station_Traces/{amp_type}/{filename}'))
        
        All_data_Traces = []
        for file in temporary_Traces:
            All_data_Traces.extend(file)
        
        All_data_UNIX = np.load(f'{data_folder}/Station_UNIX/{amp_type}/Stn{station_id}_UNIX.npy')

        return All_data_SNR, All_data_Chi, All_data_Traces, All_data_UNIX
    

    if type == 'AboveCurve_data':
        print(f'using {type}')        
        above_curve_data_SNR = np.load(f'{data_folder}/Station_SNR/{amp_type}/Stn{station_id}_SNR.npy')
        above_curve_data_Chi = np.load(f'{data_folder}/Station_Chi/{amp_type}/Stn{station_id}_Chi.npy')

        temporary_Traces = []
        for filename in os.listdir(f'{data_folder}/Station_Traces/{amp_type}/'):
            if filename.startswith(f'Stn{station_id}'):
                print(filename)
                temporary_Traces.append(np.load(f'{data_folder}/Station_Traces/{amp_type}/{filename}'))
        
        above_curve_data_Traces = []
        for file in temporary_Traces:
            above_curve_data_Traces.extend(file)

        above_curve_data_UNIX = np.load(f'{data_folder}/Station_UNIX/{amp_type}/Stn{station_id}_UNIX.npy')

        return above_curve_data_SNR, above_curve_data_Chi, above_curve_data_Traces, above_curve_data_UNIX
        

    if type == 'Circled_data':
        print(f'using {type}')
        amp_type = '200s'
        Circled_data_SNR = np.load(f'{data_folder}/Station_SNR/{amp_type}/Stn{station_id}_SNR.npy')
        Circled_data_Chi = np.load(f'{data_folder}/Station_Chi/{amp_type}/Stn{station_id}_Chi.npy')

        temporary_Traces = []
        for filename in os.listdir(f'{data_folder}/Station_Traces/{amp_type}/'):
            if filename.startswith(f'Stn{station_id}'):
                print(filename)
                temporary_Traces.append(np.load(f'{data_folder}/Station_Traces/{amp_type}/{filename}'))
        
        Circled_data_Traces = []
        for file in temporary_Traces:
            Circled_data_Traces.extend(file)

        Circled_data_UNIX = np.load(f'{data_folder}/Station_UNIX/{amp_type}/Stn{station_id}_UNIX.npy')

        return Circled_data_SNR, Circled_data_Chi, Circled_data_Traces, Circled_data_UNIX

def load_sim(path, RCR_path, backlobe_path, amp):
    RCR_files = []
    print(f'path {path + RCR_path}')
    for filename in os.listdir(path + RCR_path):
        if filename.startswith(f'FilteredSimRCR_{amp}_'):
            RCR_files.append(path + RCR_path +  filename)
    RCR = np.empty((0, 4, 256))
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
    
    # # prints out every byte in this RCR file, was printing only zeros
    # with open('/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/simulatedRCRs/200s_10.30.24/SimRCR_200s_forcedTrue_5214events_part0.npy', mode="rb") as f:
    #     data = f.read()
    #     for c in data:
    #         print(c, end = " ")

    return RCR, Backlobe

def pT(traces, title, saveLoc, sampling_rate=2, show=False, average_fft_per_channel=[]):
    # Sampling rate should be in GHz
    print(f'printing')
    # Important Clarification: In our actual experiment, we receive one data point per 0.5ns, so our duration of 128ns gives 256 data points
    # it is different from here where I organize one data point to one ns and make the total time 256ns (these two are mathematically identical)
    x = np.linspace(1, int(256 / sampling_rate), num=256)
    x_freq = np.fft.rfftfreq(len(x), d=(1 / sampling_rate * units.GHz)) / units.MHz

    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(6, 5), sharex=False)
    fmax = 0
    vmax = 0

    for chID, trace in enumerate(traces):
        trace = trace.reshape(len(trace))
        freqtrace = np.abs(fft.time2freq(trace, sampling_rate * units.GHz))

        # Plot time-domain trace
        axs[chID][0].plot(x, trace)
        
        # Plot frequency-domain trace and average FFT if provided
        if len(average_fft_per_channel) > 0:
            axs[chID][1].plot(x_freq, average_fft_per_channel[chID], color='gray', linestyle='--')
        axs[chID][1].plot(x_freq, freqtrace)

        # Update fmax and vmax for axis limits
        fmax = max(fmax, max(freqtrace))
        vmax = max(vmax, max(trace))

        # Add grid to each subplot
        axs[chID][0].grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)  # Time-domain grid
        axs[chID][1].grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)  # Frequency-domain grid

    # Set axis labels
    axs[3][0].set_xlabel('time [ns]', fontsize=12)
    axs[3][1].set_xlabel('Frequency [MHz]', fontsize=12)

    for chID, trace in enumerate(traces):
        axs[chID][0].set_ylabel(f'ch{chID}', labelpad=10, rotation=0, fontsize=10)
        axs[chID][0].set_xlim(-3, 260 / sampling_rate)
        axs[chID][1].set_xlim(-3, 1000)
        axs[chID][0].tick_params(labelsize=10)
        axs[chID][1].tick_params(labelsize=10)

        # Set y-axis limits
        axs[chID][0].set_ylim(-vmax * 1.1, vmax * 1.1)
        axs[chID][1].set_ylim(-0.05, fmax * 1.1)

    axs[0][0].tick_params(labelsize=10)
    axs[0][1].tick_params(labelsize=10)
    axs[0][0].set_ylabel(f'ch{0}', labelpad=10, rotation=0, fontsize=10)
    axs[chID][0].set_xlim(-3, 260 / sampling_rate)
    axs[chID][1].set_xlim(-3, 1000)

    fig.text(0.05, 0.5, 'Voltage [V]', ha='right', va='center', rotation='vertical', fontsize=12)
    plt.suptitle(title)

    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(left=0.175) 

    if show:
        plt.show()
    else:
        print(f'saving to {saveLoc}')
        plt.savefig(saveLoc, format='png')
    
    plt.clf()
    plt.close(fig)

    return

if __name__ == "__main__":

    # profiling method:
    start = time.time()

    end = time.time()
    print(f"task: {end - start}s")
    start = end

    # delete files:
    directory = '/pub/tangch3/ARIANNA/DeepLearning/logs'
    for i in range(154):
        files_to_delete = glob.glob(os.path.join(directory, f'Stn17_{i}.out'))

        for file in files_to_delete:
            os.remove(file)
            print(f'Deleted {file}')
