import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

import tifffile as tf
import ntpath
import time
import math
import csv
import copy
import re

from random import randint
from scipy import stats

from utils.gsheets_importer import gsheet2df, split_df, path_conversion, path_finder
from utils.paq2py import *
from utils.utils_funcs import *

import xml.etree.ElementTree as ET

import suite2p
print(suite2p.__path__)
from suite2p.run_s2p import run_s2p
from settings import ops


def experimentInfo(ss_id, sheet_name, pstation_path):
    
    date = sheet_name[0:10]

    # import and index metadata from g sheets
    df = gsheet2df(ss_id, HEADER_ROW=2, SHEET_NAME=sheet_name)

    for_processing = split_df(df, 's2p_me') # only files with TRUE in suite2p_me column
    
    if not for_processing.shape[0]:
        raise Exception('ERROR: no files set for processing')

    # at this point we have lots of files that could be whisker stim or photostim, need to find out which is which
    stim = for_processing.loc[:,'stim'] # find out what stims have been carried out
    stim_type = []
    for i,stim in enumerate(stim):
        if stim=='p': stim_type.append('p')
        if stim=='w': stim_type.append('w')
        if stim=='': stim_type.append('n')

    # get paths associated with the tiffs from each stim acquisition
    tiff_paths = for_processing.loc[:,'tiff_path']
    
    if not all(tiff_paths):
        raise Exception('ERROR: missing tiff for some entries')
    
    tiffs_pstation = []
    naparms_pstation = []
    paqs_pstation = []

    umbrella = os.path.join(pstation_path, date)
    print(umbrella)

    for i in for_processing.index:
        tiff_path = os.path.basename(for_processing.loc[i, 'tiff_path'])
        tiff_path = tiff_path.replace('"', '')
        tiff_path = path_finder(umbrella, tiff_path, is_folder=True)
        tiffs_pstation.append(tiff_path[0]) # convert paths (from Packer1 or PackerStation) to local PackerStation paths
        
        naparm_path = os.path.basename(for_processing.loc[i, 'naparm_path'])
        if naparm_path:
            naparm_path = naparm_path.replace('"', '')
            naparm_path = path_finder(umbrella, naparm_path, is_folder=True)
        else:
            naparm_path = ['none']
        naparms_pstation.append(naparm_path[0]) # convert paths (from Packer1 or PackerStation) to local PackerStation paths
        
        paq_path = os.path.basename(for_processing.loc[i, 'paq_path'])
        if paq_path:
            paq_path = paq_path.replace('"', '')
            paq_path = path_finder(umbrella, paq_path, is_folder=False)
        else:
            paq_path = ['none']
        paqs_pstation.append(paq_path[0]) # convert paths (from Packer1 or PackerStation) to local PackerStation paths
    
    return tiffs_pstation, naparms_pstation, paqs_pstation, stim_type

class interarealAnalysis():
    
    def __init__(self, ss_id, sheet_name, paths, stim):
        self.ss_id = ss_id
        self.sheet_name = sheet_name

        self.tiff_path = paths[0]
        self.naparm_path = paths[1]
        self.paq_path = paths[2]

        self.stim_type = stim
        
        self._parsePVMetadata()
    
    def _getPVStateShard(self, path, key):

        value = []
        description = []
        index = []

        xml_tree = ET.parse(path) # parse xml from a path
        root = xml_tree.getroot() # make xml tree structure

        pv_state_shard = root.find('PVStateShard') # find pv state shard element in root

        for elem in pv_state_shard: # for each element in pv state shard, find the value for the specified key

            if elem.get('key') == key: 

                if len(elem) == 0: # if the element has only one subelement
                    value = elem.get('value')
                    break

                else: # if the element has many subelements (i.e. lots of entries for that key)
                    for subelem in elem:
                        value.append(subelem.get('value'))
                        description.append(subelem.get('description'))
                        index.append(subelem.get('index'))
            else:
                for subelem in elem: # if key not in element, try subelements
                    if subelem.get('key') == key:
                        value = elem.get('value')
                        break

            if value: # if found key in subelement, break the loop
                break

        if not value: # if no value found at all, raise exception
            raise Exception('ERROR: no element or subelement with that key')

        return value, description, index

    def _parsePVMetadata(self):

        tiff_path = self.tiff_path
        path = []

        try: # look for xml file in path, or two paths up (achieved by decreasing count in while loop)
            count = 2
            while count != 0 and not path:
                count -= 1
                for dirname, dirs, files in os.walk(tiff_path):
                    for file in files:
                        if file.endswith('.xml'):
                            path = os.path.join(tiff_path, file)
                tiff_path = os.path.dirname(tiff_path)

        except:
            raise Exception('ERROR: Could not find xml for this acquisition, check it exists')

        xml_tree = ET.parse(path) # parse xml from a path
        root = xml_tree.getroot() # make xml tree structure

        sequence = root.find('Sequence')
        acq_type = sequence.get('type')

        if 'ZSeries' in acq_type:
            n_planes = len(sequence.findall('Frame'))

        else:
            n_planes = 1

        frame_period = float(self._getPVStateShard(path,'framePeriod')[0])
        fps = 1/frame_period

        frame_x = int(self._getPVStateShard(path,'pixelsPerLine')[0])

        frame_y = int(self._getPVStateShard(path,'linesPerFrame')[0])

        pixelSize, _, index = self._getPVStateShard(path,'micronsPerPixel')
        for pixelSize,index in zip(pixelSize,index):
            if index == 'XAxis':
                pix_sz_x = float(pixelSize)
            if index == 'YAxis':
                pix_sz_y = float(pixelSize)
        
        self.fps = fps
        self.frame_x = frame_x
        self.frame_y = frame_y
        self.n_planes = n_planes
        self.pix_sz_x = pix_sz_x
        self.pix_sz_y = pix_sz_y

    def s2pRun(self, user_batch_size):

        num_pixels = self.frame_x*self.frame_y
        sampling_rate = self.fps/self.n_planes
        diameter_x = 13/self.pix_sz_x
        diameter_y = 13/self.pix_sz_y
        diameter = int(diameter_x), int(diameter_y)
        batch_size = user_batch_size * (262144 / num_pixels) # larger frames will be more RAM intensive, scale user batch size based on num pixels in 512x512 images

        db = [{ 'data_path' : [self.tiff_path], 
                    'fs' : float(sampling_rate),
                    'diameter' : diameter, 
                    'batch_size' : int(batch_size), 
                    'nimg_init' : int(batch_size),
                    'nplanes' : self.n_planes
                    }]

        print(db)

        opsEnd = run_s2p(ops=ops,db=db)

    def cellAreas(self, x=None, y=None):

        self.cell_area = []

        if x:
            for i,_ in enumerate(self.cell_id):
                if self.cell_med[i][1] < x:
                    self.cell_area.append(0)
                else:
                    self.cell_area.append(1)

        if y:
            for i,_ in enumerate(self.cell_id):
                if self.cell_med[i][1] < y:
                    self.cell_area.append(0)
                else:
                    self.cell_area.append(1)

    def s2pProcessing(self):
        
        self.cell_id = []
        self.cell_plane = []
        self.cell_med = []
        self.cell_x = []
        self.cell_y = []
        self.raw = []

        for plane in range(self.n_planes):
            s2p_path = os.path.join(self.tiff_path, 'suite2p', 'plane' + str(plane))
            FminusFneu, stat = s2p_loader(s2p_path, subtract_neuropil=True)

            self.raw.append(FminusFneu)

            cell_id = []
            cell_plane = []
            cell_med = []
            cell_x = []
            cell_y = []

            for cell,s in enumerate(stat):
                cell_id.append(s['original_index']) # stat is an np array of dictionaries!
                cell_med.append(s['med'])
                cell_x.append(s['xpix'])
                cell_y.append(s['ypix'])
            
            self.cell_id.append(cell_id)
            self.cell_med.append(cell_med)
            self.cell_x.append(cell_x)
            self.cell_y.append(cell_y)

            num_units = FminusFneu.shape[0]
            cell_plane.extend([plane]*num_units)
            self.cell_plane.append(cell_plane)

    def _parseNAPARMxml(self):

        NAPARM_xml_path = path_finder(self.naparm_path, '.xml')[0]
        print(NAPARM_xml_path)

        xml_tree = ET.parse(NAPARM_xml_path)
        root = xml_tree.getroot()

        title = root.get('Name')
        n_trials = int(root.get('Iterations'))

        for elem in root:
            if int(elem[0].get('InitialDelay')) > 0:
                inter_point_delay = int(elem[0].get('InitialDelay'))

        n_groups, n_reps, n_shots = [int(s) for s in re.findall(r'\d+', title)]

        print('Numbers of trials:', n_trials, '\nNumber of groups:', n_groups, '\nNumber of shots:', n_shots, '\nNumber of sequence reps:', n_reps, '\nInter-group delay:', inter_point_delay)

        # repetitions = int(root[1].get('Repetitions'))
        # print('Repetitions:', repetitions)

        self.n_groups = n_groups
        self.n_reps = n_reps
        self.n_shots = n_shots
        self.n_trials = n_trials
        self.inter_point_delay = inter_point_delay

    def _parseNAPARMgpl(self):

        NAPARM_gpl_path = path_finder(self.naparm_path, '.gpl')[0]
        print(NAPARM_gpl_path)

        xml_tree = ET.parse(NAPARM_gpl_path)
        root = xml_tree.getroot()

        for elem in root:
            if elem.get('Duration'):
                single_stim_dur = float(elem.get('Duration'))
                print('Single stim dur (ms):', elem.get('Duration'))
                break

        self.single_stim_dur = single_stim_dur

    def paqProcessing(self):
        print(self.paq_path)
        
        paq = paq_read(self.paq_path)

        # find frame times

        clock_idx = paq['chan_names'].index('frame_clock')
        clock_voltage = paq['data'][clock_idx, :]

        frame_clock = threshold_detect(clock_voltage, 1)
        plt.figure(figsize=(10,5))
        plt.plot(clock_voltage)
        plt.plot(frame_clock, np.ones(len(frame_clock)), '.')
        sns.despine()
        plt.show()

        # find stim times

        stim_idx = paq['chan_names'].index(self.stim_channel)
        stim_volts = paq['data'][stim_idx, :]
        stim_times = threshold_detect(stim_volts, 1)

        #correct this based on txt file
        duration_ms = self.stim_dur
        frame_rate = self.fps/self.n_planes
        duration_frames = np.ceil((duration_ms/1000)*frame_rate)
        self.duration_frames = int(duration_frames)

        plt.figure(figsize=(10,5))
        plt.plot(stim_volts)
        plt.plot(stim_times, np.ones(len(stim_times)), '.')
        sns.despine()
        plt.show()

        # find stim frames

        self.stim_start_frames = []

        for plane in range(self.n_planes):
            
            stim_start_frames = []

            for stim in stim_times:

                #the index of the frame immediately preceeding stim
                stim_start_frame = next(i-1 for i,sample in enumerate(frame_clock[plane::self.n_planes]) if sample - stim >= 0)
                stim_start_frames.append(stim_start_frame)
                
            self.stim_start_frames.append(np.array(stim_start_frames))

            #sanity check
            assert max(self.stim_start_frames[0]) < self.raw[plane].shape[1]*self.n_planes

    def photostimProcessing(self):
        self._parseNAPARMxml()
        self._parseNAPARMgpl()

        single_stim = self.single_stim_dur*self.n_shots
        total_single_stim = single_stim + self.inter_point_delay 
        
        total_multi_stim = total_single_stim * self.n_groups
        
        total_stim = total_multi_stim * self.n_reps
        
        self.stim_dur = total_stim - self.inter_point_delay

        self.paqProcessing()

    def whiskerStimProcessing(self):
        self.stim_dur = 1000
        self.paqProcessing()

        self.duration_frames = 0

    def stimProcessing(self, stim_channel):
        self.stim_channel = stim_channel

        if self.stim_type == 'w':
            self.whiskerStimProcessing()
        if self.stim_type == 'p':
            self.photostimProcessing()

    def cellStaProcessing(self):
        #this is the key parameter for the sta, how many frames before and after the stim onset do you want to use
        pre_frames = int(np.ceil(self.fps*0.5)) # 500 ms pre-stim period
        post_frames = int(np.ceil(self.fps*3)) # 3000 ms post-stim period

        #list of cell pixel intensity values during each stim on each trial
        self.all_trials = [] # list 1 = cells, list 2 = trials, list 3 = dff vector

        # the average of every trial
        self.stas = [] # list 1 = cells, list 2 = sta vector

        self.all_amplitudes = []
        self.sta_amplitudes = []

        self.t_tests = []
        self.wilcoxons = []

        for plane in range(self.n_planes):

            all_trials = [] # list 1 = cells, list 2 = trials, list 3 = dff vector

            stas = [] # list 1 = cells, list 2 = sta vector

            all_amplitudes = []
            sta_amplitudes = []

            t_tests = []
            wilcoxons = []

            #loop through each cell
            for i, unit in enumerate(self.raw[plane]):

                trials = []
                amplitudes = []
                df = []
                
                # a flat list of all observations before stim occured
                pre_obs = []
                # a flat list of all observations after stim occured
                post_obs = []
                
                for stim in self.stim_start_frames[plane]:
                    
                    # get baseline values from pre_stim
                    pre_stim_f  = unit[ stim-pre_frames : stim ]
                    baseline = np.mean(pre_stim_f)

                    # the whole trial and dfof using baseline
                    trial = unit[ stim - pre_frames : stim + post_frames ]
                    trial = [ ( (f-baseline) / baseline) * 100 for f in trial ] #dff calc
                    trials.append(trial)
                    
                    #calc amplitude of response        
                    pre_f = trial[ : pre_frames ]
                    pre_f = np.mean(pre_f)
                    
                    avg_post_start = pre_frames + ( self.duration_frames + 1 )
                    avg_post_end = avg_post_start + int(np.ceil(self.fps*0.5)) # post-stim period of 500 ms
                    
                    post_f = trial[avg_post_start : avg_post_end]
                    post_f = np.mean(post_f)
                    amplitude = post_f - pre_f
                    amplitudes.append(amplitude)
                    
                    # append to flat lists
                    pre_obs.append(pre_f)
                    post_obs.append(post_f)

                    
                trials = np.array(trials)
                all_trials.append(trials)
                
                #average amplitudes across trials
                amplitudes = np.array(amplitudes)
                all_amplitudes.append(amplitudes)
                sta_amplitude = np.mean(amplitudes,0)
                sta_amplitudes.append(sta_amplitude)

                #average across all trials
                sta = np.mean(trials, 0)        
                stas.append(sta)
                
                #remove nans from flat lists
                pre_obs = [x for x in pre_obs if ~np.isnan(x)]
                post_obs = [x for x in post_obs if ~np.isnan(x)]
                
                #t_test and man whit test pre and post stim (any other test could also be used here)
                t_test = stats.ttest_rel(pre_obs, post_obs)
                t_tests.append(t_test)
                
                wilcoxon = stats.wilcoxon(pre_obs, post_obs)
                wilcoxons.append(wilcoxon)

            self.all_trials.append(np.array(all_trials))
            self.stas.append(np.array(stas))
            
            self.all_amplitudes.append(np.array(all_amplitudes))
            self.sta_amplitudes.append(np.array(sta_amplitudes))

            self.t_tests.append(np.array(t_tests))
            self.wilcoxons.append(np.array(wilcoxons))
        
        plt.figure()
        plt.plot([avg_post_start+1] * 2, [-1000, 1000])
        plt.plot([avg_post_end] * 2, [-1000, 1000])
        plt.plot([pre_frames-1] * 2, [-1000, 1000])
        plt.plot([0] * 2, [-1000, 1000])
        plt.plot(stas[5])
        plt.plot(stas[10])
        plt.plot(stas[15])
        plt.ylim([-100,200]);    

    def cellSignificance(self):
        #set this to true if you want to multiple comparisons correct for the number of cells
        multi_comp_correction = True
        if not multi_comp_correction: 
            divisor = 1
        else:
            divisor = num_units