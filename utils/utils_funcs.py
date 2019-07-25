import numpy as np
import tifffile as tf
import os
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import math

# global plotting params
params = {'legend.fontsize': 'x-large',
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
plt.rcParams.update(params)
sns.set()
sns.set_style('white')


def dfof(arr):

    '''takes 1d list or array or 2d array and returns dfof array of same
       dim (JR 2019) This is extraordinarily slow, use dfof2'''


    if type(arr) is list or type(arr) == np.ndarray and len(arr.shape) == 1:
        F = np.mean(arr)
        dfof_arr = [((f - F) / F) * 100 for f in arr]

    elif type(arr) == np.ndarray and len(arr.shape) == 2:
        dfof_arr = []
        for trace in arr:
            F = np.mean(trace)
            dfof_arr.append([((f - F) / F) * 100 for f in trace])

    else:
        raise NotImplementedError('input type not recognised')

    return np.array(dfof_arr)


def dfof2(flu):
    '''
    delta f over f, this function is orders of magnitude faster than the dumb one above
    takes input matrix flu (num_cells x num_frames)
    (JR 2019)

    '''

    flu_mean = np.mean(flu,1)
    flu_mean = np.reshape(flu_mean, (len(flu_mean), 1))
    return (flu - flu_mean) / flu_mean


def get_tiffs(path):

    tiff_files = []
    for file in os.listdir(path):
        if file.endswith('.tif') or file.endswith('.tiff'):
            tiff_files.append(os.path.join(path, file))

    return tiff_files


def s2p_loader(s2p_path, subtract_neuropil=True, neuropil_coeff = 0.7):

    found_stat = False

    for root, dirs, files in os.walk(s2p_path):

        for file in files:

            if file == 'F.npy':
                all_cells = np.load(os.path.join(root, file))
            elif file == 'Fneu.npy':
                neuropil = np.load(os.path.join(root, file))
            elif file == 'iscell.npy':
                is_cells = np.load(os.path.join(root, file))[:, 0]
                is_cells = np.ndarray.astype(is_cells, 'bool')
                print('Loading {} traces labelled as cells'.format(sum(is_cells)))
            elif file == 'stat.npy':
                stat = np.load(os.path.join(root, file))
                found_stat = True

    if not found_stat:
        raise FileNotFoundError('Could not find stat, ' \
                                'this is likely not a suit2p folder')
    for i, s in enumerate(stat):
        s['original_index'] = i

    stat = stat[is_cells]

    if not subtract_neuropil:
        return all_cells[is_cells, :], stat

    else:
        print('Subtracting neuropil with a coefficient of {}'.format(neuropil_coeff))
        neuropil_corrected = all_cells - neuropil * neuropil_coeff
        return neuropil_corrected[is_cells, :], stat


def correct_s2p_combined(s2p_path):

    len_count = 0 
    for i in range(3):
        iscell = np.load(os.path.join(s2p_path, 'plane{}'.format(i), 'iscell.npy'))
        if i == 0: 
            allcells = iscell
        else: 
            allcells = np.vstack((allcells, iscell))

        len_count += len(iscell)
        
    combined_iscell = os.path.join(s2p_path, 'combined', 'iscell.npy')

    ic =  np.load(combined_iscell)
    assert ic.shape == allcells.shape
    assert len_count == len(ic)

    np.save(combined_iscell, allcells)



def read_fiji(csv_path):

    '''reads the csv file saved through plot z axis profile in fiji'''

    data = []

    with open(csv_path, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for i, row in enumerate(spamreader):
            if i == 0:
                continue
            data.append(float(row[0].split(',')[1]))

    return np.array(data)


def save_fiji(arr):
    '''saves numpy array in current folder as fiji friendly tiff'''
    tf.imsave('Vape_array.tiff', arr.astype('int16'))


def threshold_detect(signal, threshold):
    '''lloyd russell'''
    thresh_signal = signal > threshold
    thresh_signal[1:][thresh_signal[:-1] & thresh_signal[1:]] = False
    times = np.where(thresh_signal)
    return times[0]


def pade_approx_norminv(p):
    q = math.sqrt(2*math.pi) * (p - 1/2) - (157/231) * math.sqrt(2) * \
        math.pi**(3/2) * (p - 1/2)**3
    r = 1 - (78/77) * math.pi * (p - 1/2)**2 + (241 * math.pi**2 / 2310) * \
        (p - 1/2)**4
    return q/r


def d_prime(hit_rate, false_alarm_rate):
    return pade_approx_norminv(hit_rate) - \
           pade_approx_norminv(false_alarm_rate)


def paq_data(paq, chan_name, threshold_ttl=False, plot=False):
    '''
    returns the data in paq (from paq_read) from channel: chan_names
    if threshold_tll: returns sample that trigger occured on
    '''

    chan_idx = paq['chan_names'].index(chan_name)
    data = paq['data'][chan_idx, :]
    if threshold_ttl:
        data = threshold_detect(data, 1)

    if plot:
        if threshold_ttl:
            plt.plot(data, np.ones(len(data)), '.')
        else:
            plt.plot(data)

    return data


def stim_start_frame(paq, stim_chan_name):

    '''gets the frames (from channel frame_clock) that a stim occured on'''

    frame_clock = paq_data(paq, 'frame_clock', threshold_ttl=True)
    stim_times = paq_data(paq, stim_chan_name, threshold_ttl=True)

    stim_times = [stim for stim in stim_times if stim < max(frame_clock)]

    frames = []

    for stim in stim_times:
        # the sample time of the frame immediately preceeding stim
        frame = next(frame_clock[i-1] for i, sample in enumerate(frame_clock)
                     if sample - stim > 0)
        frames.append(frame)

    return(frames)


def myround(x, base=5):

    '''allow rounding to nearest base number for
       use with multiplane stack slicing'''

    return base * round(x/base)




