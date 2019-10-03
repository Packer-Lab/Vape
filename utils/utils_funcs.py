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
            elif file == 'spks.npy':
                spks = np.load(os.path.join(root, file))
            elif file == 'stat.npy':
                stat = np.load(os.path.join(root, file))
                found_stat = True

    if not found_stat:
        raise FileNotFoundError('Could not find stat, ' \
                                'this is likely not a suit2p folder')
    for i, s in enumerate(stat):
        s['original_index'] = i

    stat = stat[is_cells]

    spks = spks[is_cells,:]

    if not subtract_neuropil:
        return all_cells[is_cells, :],  spks, stat


    else:
        print('Subtracting neuropil with a coefficient of {}'.format(neuropil_coeff))
        neuropil_corrected = all_cells - neuropil * neuropil_coeff
        return neuropil_corrected[is_cells, :], spks, stat


def correct_s2p_combined(s2p_path, n_planes):

    len_count = 0 
    for i in range(n_planes):

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


def stim_start_frame(paq=None, stim_chan_name=None, frame_clock=None, 
                     stim_times=None):

    '''Returns the frames from a frame_clock that a stim occured on.
       Either give paq and stim_chan_name as arugments if using 
       unprocessed paq. 
       Or predigitised frame_clock and stim_times in reference frame
       of that clock
    
    '''
 
    if frame_clock is None:
        frame_clock = paq_data(paq, 'frame_clock', threshold_ttl=True)
        stim_times = paq_data(paq, stim_chan_name, threshold_ttl=True)

    stim_times = [stim for stim in stim_times if stim < np.nanmax(frame_clock)]

    frames = []

    for stim in stim_times:
        # the sample time of the frame immediately preceeding stim
        frame = next(frame_clock[i-1] for i, sample in enumerate(frame_clock)
                     if sample - stim > 0)
        frames.append(frame)

    return np.array(frames)


def myround(x, base=5):

    '''allow rounding to nearest base number for
       use with multiplane stack slicing'''

    return base * round(x/base)


def tseries_finder(tseries_lens, frame_clock, paq_rate=20000):

    '''Finds chunks of frame clock that correspond to the tseries in tseries lens

       tseries_lens -- list of the number of frames each tseries you want to find 
                       contains
       frame_clock  -- thresholded times each frame recorded in paqio occured
       
       paq_rate     -- input sampling rate of paqio

       '''

    # frame clock recorded in paqio, includes TTLs from cliking 'live' and foxy extras 
    clock = frame_clock / paq_rate 

    # break the recorded frame clock up into individual aquisitions
    # where TTLs are seperated by more than 1s
    gap_idx = np.where(np.diff(clock) > 1)
    gap_idx = np.insert(gap_idx, 0, 0)
    gap_idx = np.append(gap_idx, len(clock))
    chunked_paqio = np.diff(gap_idx)

    # are each of the frames recorded by the frame clock actually in processed tseries?
    real_frames = np.zeros(len(clock))
    # the max number of extra frames foxy could spit out
    foxy_limit = 20
    # the number of tseries blocks that have already been found
    series_found = 0 
    # count how many frames have been labelled as real or not
    counter = 0 

    for chunk in chunked_paqio:
        is_tseries = False
        
        # iterate through the actual length of each analysed tseries
        for idx, ts in enumerate(tseries_lens):
            # ignore previously found tseries
            if idx < series_found: 
                continue
                
            # the frame clock block matches the number of frames in a tseries
            if chunk >= ts and chunk <= ts + foxy_limit:            
                # this chunk of paqio clock is a recorded tseries
                is_tseries = True            
                # advance the number of tseries found so they are not detected twice 
                series_found +=1      
                break
        
        if is_tseries:
            # foxy bonus frames
            extra_frames = chunk - ts
            # mark tseries frames as real
            real_frames[counter:counter+ts] = 1
            # move the counter on by the length of the real tseries
            counter += ts
            # set foxy bonus frames to not real
            real_frames[counter:counter+extra_frames] = 0
            # move the counter along by the number of foxy bonus frames
            counter += extra_frames
            
        else:
            # not a tseries so just move the counter along by the chunk of paqio clock
            counter += chunk  + 1 # this could be wrong, not sure if i've fixed the ob1 error, go careful
                
    real_idx = np.where(real_frames==1)

    return frame_clock[real_idx]



def flu_splitter(flu, clock, t_starts, pre_frames, post_frames):

    '''Split a fluoresence matrix into trial by trial array

       flu -- fluoresence matrix [num_cells x num_frames]
       clock -- the time that each frame occured
       t_starts -- the time each frame started
       pre_frames -- the number of frames before t_start
                     to include in the trial
       post_frames --  the number of frames after t_start
                       to include in the trial
       
       returns 
       trial_flu -- trial by trial array 
                    [num_cells x trial frames x num_trials]
       imaging_trial -- list of booleans of len num_trials,
                        was the trial imaged? 

       n.b. clock and t_start must have same units and
            reference frame (see rsync_aligner.py

       '''

    assert flu.shape[1] == len(clock), '{} frames in fluorescent array '\
                                       '{} frames in clock'\
                                       .format(flu.shape[1], len(clock))
    
    num_trials = len(t_starts)
    imaging_trial = [False] * num_trials # was the trial imaged?
    
    first = True # dumb way of building array in for loop
    for trial, t_start in enumerate(t_starts):
        # the trial occured before imaging started 
        if t_start < min(clock): 
            continue
            
        # find the first frame occuring after each trial start
        for idx, frame in enumerate(clock):
            # the idx the frame immediiately proceeds the t_start
            if frame - t_start >= 0:
            
                imaging_trial[trial] = True
                flu_chunk = flu[:,idx-pre_frames:idx+post_frames]
                
                if first:
                    trial_flu = flu_chunk
                    first = False
                else:
                    trial_flu = np.dstack((trial_flu, flu_chunk))                
                break 

    assert trial_flu.shape[2] == sum(imaging_trial)

    return trial_flu, imaging_trial


def closest_frame_before(clock, t):
    ''' returns the idx of the frame immediately preceeding 
        the time t. Frame clock must be digitised and expressed
        in the same reference frame as t
        '''
    subbed = np.array(clock) - t
    return np.where(subbed < 0, subbed, -np.inf).argmax()


def raster_plot(arr, y_pos=1, color=np.random.rand(3,), alpha=1,
                marker='.', markersize=12):

    plt.plot(arr, np.ones(len(arr)) * y_pos, marker,
            color=color, alpha=alpha, markersize=markersize)
            







