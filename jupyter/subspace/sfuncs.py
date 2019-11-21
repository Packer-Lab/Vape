import numpy as np
import utils.utils_funcs as utils

def stim_start_frame(paq, stim_chan_name):

    '''gets the frames (from channel frame_clock) that a stim occured on
        
    '''

    frame_clock = utils.paq_data(paq, 'frame_clock', threshold_ttl=True)
    stim_times = utils.paq_data(paq, stim_chan_name, threshold_ttl=True)

    stim_times = [stim for stim in stim_times if stim < max(frame_clock)]

    frames = []

    for stim in stim_times:
        # the sample time of the frame immediately preceeding stim
        frame = next(i-1 for i, sample in enumerate(frame_clock) if sample - stim >= 0)
        frames.append(frame)    
    return(frames)

def stim_start_samples(paq, stim_chan_name):

    '''gets the sample number (from channel frame_clock) that a stim occured on'''

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
    
def getIndicesFromStat_1plane(stat,conditional=None):
    #retrieve the indices where a condition is met
    #planes is list of planes [0,1,2] zero indexed
    #conditional is a tuple with ('key',bool or other value). Only indices of cells that meet the condition key==value will be appended.
    planelist = []
    if conditional == None:
        print('Retrieving indices unconditionally')
        for i, cell in enumerate(stat):
            planelist.append(i)
    else:
        print('Retrieving indices for cells where condition met:',conditional[0],'==',conditional[1])
        for i, cell in enumerate(stat):
            if stat[i][conditional[0]] == conditional[1]:
                planelist.append(i)          
    return(planelist)

def getKeyFromStat_1plane(stat,key,conditional=None):
    #retrieve the values inside a dictionary key of a stat file, append to list
    #planes is list of planes [0,1,2] zero indexed
    #conditional is a tuple with ('key',bool or other value). Only keys that meet the condition key==value will be appended.
    # e.g. if you want the S2P indices of significant cells, run: getKeyFromStat(RL042_S1_stat,[0,1,2],'original_index',('issig',1))

    listy = []
    if conditional == None:
        print('Retrieving values for',key,'key unconditionally')
        for i, cell in enumerate(stat):
            planelist.append(stat[i][key])

    else:
        print('Retrieving values for',key,'key where condition met:',conditional[0],'==',conditional[1])
        for i, cell in enumerate(stat):
            if stat[i][conditional[0]] == conditional[1]:
                listy.append(stat[i][key])
    return(listy)


def separateByX(stat,xcoord):
    s1_indices = []
    s2_indices = []
    for i, cell in enumerate(stat):
        medX = stat[i]['med']
        if medX[1]<= xcoord:
            s1_indices.append(i)
        if medX[1]> xcoord:
            s2_indices.append(i)
    return(s1_indices,s2_indices)