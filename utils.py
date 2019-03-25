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
    thresh_signal = signal > threshold
    thresh_signal[1:][thresh_signal[:-1] & thresh_signal[1:]] = False
    times = np.where(thresh_signal)
    return times[0]



 
def process_stim_artefact(stack=None, db=None, threshold=1.5, interpolate=True): 
    '''  
    remove frames with mean pixel intensity higher than 
    threhold x average across time series
    returns stack with linear interpolation through these stim frames 
    
    takes input of stack OR db
    
    if stack: returns artifact cleaned stack array
    if db: returns db dict with path now to cleaned tiff (JR 2019) 
    
    interpolate: whether to interpolate through removed frames or replace with 0s
    '''

    if not stack and not db: raise ValueError('must pass function stack or db')
    if stack and db: raise ValueError('do not pass function stack and db')
       
    if db: 
        tiff_files = get_tiffs(db['data_path'][0])
        
        if len(tiff_files) > 1: raise NotImplementedError('can only support single tiff file in folder currently')
        tiff_file = tiff_files[0]
        stack = tf.imread(tiff_file)

    dims = stack.shape
    n_frames = dims[0]
    av_frame = np.mean(stack, axis=(1,2))
    
    print('calculating frames to remove')
    # Frames with averge fluoresence threshold * higher than average
    to_remove = np.where(av_frame > threshold*np.mean(av_frame))[0]

    # list of frames not blanked
    xt = np.arange(n_frames)
    xt = np.delete(xt,to_remove)

    if interpolate:
        print('interpolating through stim frames')
        
        #remove the frames that are above threshold
        blanked = np.delete(stack, to_remove, axis=0)
        
        #perform pixel wise linear interpolation across blanked frames
        for row in range(dims[1]):
            for col in range(dims[2]):
                px = blanked[:,row,col]
                intp = np.interp(to_remove, xt, px)
                stack[to_remove,row,col] = intp
           
    else:
        print('setting stim frames to 0')
        stack[to_remove, :, :] = 0
   
    assert stack.shape == dims
    
    if db:    
        #update db path
        print('updating db to link to stim removed tiff')
        ar_path = os.path.join(os.path.dirname(tiff_file), 'artifactRemoved')            
        db['data_path'] = [ar_path]   
        if not os.path.exists(ar_path): os.makedirs(ar_path)
         
        #write artifact removed stack
        exp_name = ntpath.basename(tiff_file).split('.')[0]
        output_path = os.path.join(ar_path, exp_name +'_artifactRemoved.tiff')
        print('saving tiff')
        tf.imwrite(output_path, stack, photometric='minisblack')
       
        return db
    
    else:
        return stack



