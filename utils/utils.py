import numpy as np
import tifffile as tf
import ntpath
import os
import csv
from lxml import objectify
from lxml import etree



   
def dfof(arr):

    '''takes 1d list or array or 2d array and returns dfof array of same dim (JR 2019)'''
       
    if type(arr) is list or type(arr) == np.ndarray and len(arr.shape) == 1:
        F = np.mean(arr)
        dfof_arr = [((f- F) / F) * 100 for f in arr]
        
    elif type(arr) == np.ndarray and len(arr.shape) == 2:
        dfof_arr = []
        for trace in arr:
            F = np.mean(trace)
            dfof_arr.append([((f - F) / F) * 100 for f in trace])
            
    else:
        raise NotImplementedError('input type not recognised')
        
    return np.array(dfof_arr)
    
       

def get_tiffs(path):
    
    tiff_files = []
    for file in os.listdir(path):
        if file.endswith('.tif') or file.endswith('.tiff'):
            tiff_files.append(os.path.join(path,file))
                   
    return tiff_files
    
    
def s2p_loader(s2p_path, subtract_neuropil=True):

    for root,dirs,files in os.walk(s2p_path):

        for file in files:

            if file == 'F.npy':
                all_cells = np.load(os.path.join(root, file))
            elif file == 'Fneu.npy':
                neuropil = np.load(os.path.join(root, file))
            elif file == 'iscell.npy':
                is_cells = np.load(os.path.join(root, file))[:,0]
                is_cells = np.ndarray.astype(is_cells, 'bool')   
            elif file == 'stat.npy':
                stat = np.load(os.path.join(root, file))
                                

    for i,s in enumerate(stat):
        s['original_index'] = i
        
    stat = stat[is_cells] 

    if not subtract_neuropil:
        return all_cells[is_cells, :], stat
    
    else:
        neuropil_corrected = all_cells - neuropil
        return neuropil_corrected[is_cells, :], stat
        
        
        
def read_fiji(csv_path):

    '''reads the csv file saved through plot z axis profile in fiji'''
    
    data = []
    
    with open(csv_path, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for i,row in enumerate(spamreader):
            if i ==0 : continue
            data.append(float(row[0].split(',')[1]))

    return np.array(data)

    
def save_fiji(arr):
    '''saves numpy array in current folder as fiji friendly tiff'''
    tf.imsave('Vape_array.tiff', arr.astype('int16'))

        
def digitise_trigger(volts, trig_gradient=0.1, min_time=0, max_time=np.Inf):
    
    '''
    inputs 
    
    volts:         analogue voltage trace containing triggers
    trig_gradient: +ve voltage gradient requried to register trigger (0.1 V works well with packio triggers)
    min_time:      two triggers should not occur within the time frame
    max_time:      max time gap expected between two triggers (set np.Inf if not sure)
     
    returns:       trigger times in samples
    
    '''   
    # sample indexes of a rising voltage
    # should not be more than a 0.1 V change when there is no trigger.
    samples =  np.where(np.diff(volts) > trig_gradient)[0]

    # time difference between voltage upstrokes
    upstroke_diff = np.diff(samples)

    # time difference between all triggers should be greater than min_time but less than max_time
    filter_idx = np.where((upstroke_diff > min_time) & (upstroke_diff < max_time))[0]

    #hack to get the last trigger, this would break if there are > 1 0.1V diff voltages recorded on
    #penultimate trigger, this is unlikely, though change in future
    filter_idx = np.append(filter_idx, filter_idx[-1] + 1)

    return samples[filter_idx]
    
    
def threshold_detect(signal, threshold):
    '''lloyd russell'''
    thresh_signal = signal > threshold
    thresh_signal[1:][thresh_signal[:-1] & thresh_signal[1:]] = False
    times = np.where(thresh_signal)
    return times[0]


        





