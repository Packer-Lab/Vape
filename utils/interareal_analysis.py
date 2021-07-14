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
import pickle
import ntpath
import bisect

from random import randint
from scipy import stats, signal
import statsmodels.stats as smstats

from utils.gsheets_importer import gsheet2df, split_df, path_conversion, path_finder
from utils.paq2py import *
from utils.utils_funcs import *

import xml.etree.ElementTree as ET

from suite2p.run_s2p import run_s2p

from settings import ops



class interarealProcessing():

    
    def __init__(self, ss_id, sheet_name, pstation_path):
        '''
        Makes an object for the entire experimental session (session object)
        
        Inputs:
            ss_id            - Google spreadsheet ID
            sheet_name       - the name of the sheet in the Google spreadsheet
            pstation_path    - data storage server path
        Attributes:
            s2p_path         - path to suite2p data
            photostim_s      - experimental object
            photostim_r      - experimental object
            whisker_stim     - experimental object
            spont            - experimental object
        Methods:
            s2pRun           - run suite2p on the data from this experimental session
            getFrameRanges   - get the range of frames for each experiment
            addShamPhotostim - add photostim metadata to the spont experimental object
            whiskerTargets   - whether or not targeted, whisker-responsive cells responded on each trial 
        '''
        self.ss_id = ss_id
        self.sheet_name = sheet_name
        print('\n================================================================')
        print('Fetching paths and stim types for:', self.sheet_name)
        print('================================================================')
        
        sorted_paths = self._getDataPaths(pstation_path) # data paths

        self.s2p_path = os.path.join(sorted_paths[0][0], 'suite2p', 'plane0')

        self.photostim_r = interarealAnalysis(sorted_paths[0], 'markpoints2packio', 'pr', self.s2p_path, self.sheet_name)
        self.photostim_s = interarealAnalysis(sorted_paths[1], 'markpoints2packio', 'ps', self.s2p_path, self.sheet_name)
        self.whisker_stim = interarealAnalysis(sorted_paths[2], 'piezo_stim', 'w', self.s2p_path, self.sheet_name)
        self.spont = interarealAnalysis(sorted_paths[3], 'markpoints2packio', 'none', self.s2p_path, self.sheet_name)

        
    def _sortPaths(self, tiffs_pstation, naparms_pstation, paqs_pstation, stim_type):
        '''
        Link the paths to their appropriate experiment within this session
        
        Inputs: 
            tiffs_pstation    - tiff folder paths on storage server
            naparms_pstation  - naparm folder paths on storage server
            paqs_pstation     - paq paths on storage server
            stim_type         - what stimulation happened during this experiment
        Outputs:
            photostim_sensory - numpy array of data paths
            photostim_random  - ""
            whisker_stim      - ""
            spontaneous       - ""
        '''
        pr_id = np.where(stim_type=='pr') #photostim random groups
        ps_id = np.where(stim_type=='ps') #photostim similar groups
        w_id = np.where(stim_type=='w') #whisker stim
        n_id = np.where(stim_type=='n') #no stim (spontaneous)
        
        photostim_sensory = np.concatenate((tiffs_pstation[ps_id], naparms_pstation[ps_id], paqs_pstation[ps_id]))
        photostim_random = np.concatenate((tiffs_pstation[pr_id], naparms_pstation[pr_id], paqs_pstation[pr_id]))
        whisker_stim = np.concatenate((tiffs_pstation[w_id], naparms_pstation[w_id], paqs_pstation[w_id]))
        spontaneous = np.concatenate((tiffs_pstation[n_id], naparms_pstation[n_id], paqs_pstation[n_id]))

        return photostim_random, photostim_sensory, whisker_stim, spontaneous

    
    def _getDataPaths(self, pstation_path):
        '''
        Retrieve and process data paths in to data storage server paths
        
        Inputs:
            pstation_path - path to data storage server
        Outputs:
            sorted_paths  - array of data paths linked to stimulation types 
        '''
        # date for this experiment (assuming all sheet names start with yyyy-mm-dd)
        date = self.sheet_name[0:10]

        # import and index metadata from g sheets as a Pandas dataframe
        df = gsheet2df(self.ss_id, HEADER_ROW=2, SHEET_NAME=self.sheet_name)

        for_processing = split_df(df, 's2p_me') # only rows with TRUE in suite2p_me column
        
        if not for_processing.shape[0]:
            raise Exception('ERROR: no files set for processing')

        # find out which stims were performed in each row
        stim_list = for_processing.loc[:,'stim']
        stim_type = []
        for stim in stim_list:
            if stim=='pr': stim_type.append('pr')
            if stim=='ps': stim_type.append('ps')
            if stim=='w': stim_type.append('w')
            if stim=='': stim_type.append('n')

        tiff_paths = for_processing.loc[:,'tiff_path']
        
        if not all(tiff_paths):
            raise Exception('ERROR: missing tiff for some entries')
        
        tiffs_pstation = []
        naparms_pstation = []
        paqs_pstation = []

        umbrella = os.path.join(pstation_path, date)
        print('\nUmbrella folder:', umbrella)
        
        # convert paths from experimental PC to data storage server paths
        for i in for_processing.index:
            tiff_path = for_processing.loc[i, 'tiff_path'].replace('"', '')
            tiff_path = ntpath.basename(tiff_path)
            tiff_path = path_finder(umbrella, tiff_path, is_folder=True)
            tiffs_pstation.append(tiff_path[0]) 
            
            naparm_path = for_processing.loc[i, 'naparm_path'].replace('"', '')
            naparm_path = ntpath.basename(naparm_path)
            if naparm_path:
                naparm_path = path_finder(umbrella, naparm_path, is_folder=True)
            else:
                naparm_path = ['none']
            naparms_pstation.append(naparm_path[0])
            
            paq_path = for_processing.loc[i, 'paq_path'].replace('"', '')
            paq_path = ntpath.basename(paq_path)
            if paq_path:
                paq_path = path_finder(umbrella, paq_path, is_folder=False)
            else:
                paq_path = ['none']
            paqs_pstation.append(paq_path[0])
        
        # make lists in to arrays
        tiffs_pstation = np.array(tiffs_pstation)
        naparms_pstation = np.array(naparms_pstation)
        paqs_pstation = np.array(paqs_pstation)
        stim_type = np.array(stim_type)
        
        # link the paths with the stim types
        sorted_paths = self._sortPaths(tiffs_pstation, naparms_pstation, paqs_pstation, stim_type)
        print('\nExperimental sorted_paths =', sorted_paths)

        return sorted_paths
    
    
    def _getPhotostimFrames(self, obj):
        '''
        Find the exact frame numbers where photostimulation occurred
        
        Inputs:
            obj              - experimental object
        Outputs:
            photostim_frames - list of frame indexes on which photostim occurred
        '''
        photostim_frames = []

        stim_duration_frames = list(range(0,obj.duration_frames+1)) 
        
        for frame in obj.stim_start_frames[0]:
            new_photostim_frames = frame + stim_duration_frames
            photostim_frames.extend(new_photostim_frames)
            
        return photostim_frames
    
    
    def s2pRun(self, tau):
        '''
        Initiate Suite2p run for this experimental session, setting custom parameters
        according to the metadata for this session
        '''
        # customise parameters for this session
        sampling_rate = self.photostim_r.fps/self.photostim_r.n_planes
        diameter_x = 13/self.photostim_r.pix_sz_x
        diameter_y = 13/self.photostim_r.pix_sz_y
        diameter = int(diameter_x), int(diameter_y)
        n_planes = self.photostim_r.n_planes
        
        # make list of data paths n.b. photostim_r always put first
        data_folders = [
                        self.photostim_r.tiff_path, 
                        self.photostim_s.tiff_path,
                        ]

        if self.whisker_stim.tiff_path:
            data_folders.extend([self.whisker_stim.tiff_path])

        if self.spont.tiff_path:
            data_folders.extend([self.spont.tiff_path])

        # find which frames the stimulation occurred on in photostim experiments
        # this must be called 'bad_frames.npy' for suite2p to read it, and from only
        # the first data folder (photostim_r)
        photostim_r_frames = np.array(self._getPhotostimFrames(self.photostim_r))
        photostim_s_frames = np.array(self._getPhotostimFrames(self.photostim_s)) + self.photostim_r.n_frames
        photostim_frames = np.concatenate((photostim_r_frames, photostim_s_frames))
        first_data_folder = self.photostim_r.tiff_path
        np.save(os.path.join(first_data_folder, 'bad_frames.npy'), photostim_frames) 

        db = {
            'data_path' : data_folders, 
            'fs'        : float(sampling_rate),
            'diameter'  : diameter, 
            'nplanes'   : n_planes,
            'tau'       : tau
            }

        print('\ns2p ops:', db)

        run_s2p(ops=ops,db=db)
    
    
    def getFrameRanges(self):
        '''
        Find the range and total number of frames for each experiment
        '''
        
        print('\nCalculating frame ranges...')
        
        s2p_path = self.s2p_path
        ops_path = os.path.join(s2p_path, 'ops.npy')
        ops = np.load(ops_path, allow_pickle=True)
        ops = ops.item()

        frame_list = ops['frames_per_folder']
        self.frame_list = frame_list
        
        # find total number of frames
        i=0

        self.photostim_r.n_frames = frame_list[i]
        i+=1

        self.photostim_s.n_frames = frame_list[i]
        i+=1

        if self.whisker_stim.n_frames > 0: 
            self.whisker_stim.n_frames = frame_list[i]
            i+=1
            
        if self.spont.n_frames > 0: 
            self.spont.n_frames = frame_list[i]
            i+=1
        
        # find range of frames
        self.photostim_r.frames = range(0,self.photostim_r.n_frames)
        subtotal = self.photostim_r.n_frames
        self.photostim_s.frames = range(subtotal,self.photostim_s.n_frames+subtotal)
        subtotal += self.photostim_s.n_frames
        self.whisker_stim.frames = range(subtotal,self.whisker_stim.n_frames+subtotal)
        subtotal += self.whisker_stim.n_frames
        self.spont.frames = range(subtotal,self.spont.n_frames+subtotal)
        
        
    def addShamPhotostim(self):
        '''
        Add photostim data to spontaneous experiments to create a sham experiment
        (only necessary if no sham experiment performed)
        '''
        print('\n----------------------------------------------------------------')
        print('Obtaining sham metadata for spont:', self.spont.tiff_path)
        print('----------------------------------------------------------------')
        
        if self.spont.naparm_path == 'none':
            self.spont.stim_start_frames = self.photostim_r.stim_start_frames
            self.spont.naparm_path = self.photostim_r.naparm_path
            self.spont.spiral_size = self.photostim_r.spiral_size 
            self.spont.duration_frames = self.photostim_r.duration_frames
            self.spont.stim_dur = self.photostim_r.stim_dur
            self.spont.stim_freq = self.photostim_r.stim_freq
            self.spont.single_stim_dur = self.photostim_r.single_stim_dur
            self.spont.n_reps = self.photostim_r.n_reps
            self.spont.n_shots = self.photostim_r.n_shots
            self.spont.n_groups = self.photostim_r.n_groups
            self.spont.n_trials = self.photostim_r.n_trials
            self.spont.inter_point_delay = self.photostim_r.inter_point_delay
            self.spont.targeted_cells = self.photostim_r.targeted_cells
            self.spont.n_targets = self.photostim_r.n_targets
            self.spont.n_targeted_cells = self.photostim_r.n_targeted_cells
            self.spont.sta_euclid_dist = self.photostim_r.sta_euclid_dist
        else: 
            self.spont.photostimProcessing()
            self.spont.n_trials = self.photostim_r.n_trials 
            self.spont.targeted_cells = self.photostim_r.targeted_cells
            self.spont.n_targets = self.photostim_r.n_targets

        
    def _targetedWhiskerCells(self, exp_obj):
        '''
        For each trial, find cells that were whisker-responsive, were targeted and trial-responsive
        
        Input:
            exp_obj               - experimental object (i.e. photostim_r, photostim_s)
        Output:
            targeted_w_responders - cells that were whisker-responsive, targeted and trial-responsive [cells x trials]
        '''
        trial_responders = exp_obj.trial_sig_dff[0]

        targeted_cells = np.repeat(exp_obj.targeted_cells[..., None], 
                                   trial_responders.shape[1], 1) # [..., None] is a quick way to expand_dims

        whisker_cells = np.repeat(self.whisker_stim.sta_sig[0][..., None],
                                  trial_responders.shape[1], 1)

        targeted_w_responders = targeted_cells & whisker_cells & trial_responders

        return targeted_w_responders
    
    
    def whiskerTargets(self):
        '''
        If a whisker stim was done, find which cells were targeted, whisker-responsive and trial-responsive
        '''
        if self.whisker_stim.n_frames > 0:
            self.photostim_r.trial_w_targets = self._targetedWhiskerCells(self.photostim_r)
            self.photostim_s.trial_w_targets = self._targetedWhiskerCells(self.photostim_s)
            self.spont.trial_w_targets = self._targetedWhiskerCells(self.spont)
    
    
        
class interarealAnalysis():

    
    def __init__(self, sorted_paths, stim_channel, stim_type, s2p_path, sheet_name):
        '''
        Makes an object for a single recording in a session (experiment object)
        
        Inputs:
            sorted_paths - data paths organised by stim type (from parent object)
            stim_channel - string from PackIO
            stim_type    - string of the stimulus type for this experiment
            s2p_path     - path to suite2p files
            sheet_name   - sheet name from Google spreadsheet
        Methods:
            s2pAnalysis  - calculate trial-wise and average metrics for each cell for each stim type
        '''
        if any(sorted_paths):            
            self.tiff_path = sorted_paths[0]
            self.naparm_path = sorted_paths[1]
            self.paq_path = sorted_paths[2]
            
            self.sheet_name = sheet_name
            self.s2p_path = s2p_path
            self.stim_channel = stim_channel
            self.stim_type = stim_type

            print('\n----------------------------------------------------------------')
            print('Obtaining metadata for', self.stim_type, 'stim:', self.tiff_path)
            print('----------------------------------------------------------------')
            
            self._parsePVMetadata()
                        
            self.stimProcessing()
        
        else:
            self.stim_type = stim_type
            
            self.tiff_path = None
            self.naparm_path = None
            self.paq_path = None

            self.n_frames = 0
            
            print('\n----------------------------------------------------------------')
            print('No metadata for this', self.stim_type, 'stim experiment')
            print('----------------------------------------------------------------')
    
    
    def _getPVStateShard(self, root, key):
        '''
        Find the value, description and indices of a particular parameter from an xml file
        
        Inputs:
            path        - path to xml file
            key         - string corresponding to key in xml tree
        Outputs:
            value       - value of the key
            description - unused
            index       - index that the key was found at
        '''
        value = []
        description = []
        index = []

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
        '''
        Parse all of the relevant acquisition metadata from the PrairieView xml file for this recording
        '''
        tiff_path = self.tiff_path # starting path
        xml_path = [] # searching for xml path
        
        try: # look for xml file in path, or two paths up (achieved by decreasing count in while loop)
            count = 2
            while count != 0 and not xml_path:
                count -= 1
                for file in os.listdir(tiff_path):
                    if file.endswith('.xml'):
                        xml_path = os.path.join(tiff_path, file)
                tiff_path = os.path.dirname(tiff_path) # re-assign tiff_path as next folder up

        except:
            raise Exception('ERROR: Could not find xml for this acquisition, check it exists')

        xml_tree = ET.parse(xml_path) # parse xml from a path
        root = xml_tree.getroot() # make xml tree structure

        sequence = root.find('Sequence')
        acq_type = sequence.get('type')

        if 'ZSeries' in acq_type:
            n_planes = len(sequence.findall('Frame'))
        else:
            n_planes = 1
        
        frame_branch = root.findall('Sequence/Frame')[-1]
#         frame_period = float(self._getPVStateShard(root,'framePeriod')[0])
        frame_period = float(self._getPVStateShard(frame_branch, 'framePeriod')[0])
        fps = 1/frame_period
        
        frame_x = int(self._getPVStateShard(root,'pixelsPerLine')[0])
        frame_y = int(self._getPVStateShard(root,'linesPerFrame')[0])
        zoom = float(self._getPVStateShard(root,'opticalZoom')[0])

        scan_volts, _, index = self._getPVStateShard(root,'currentScanCenter')
        for scan_volts,index in zip(scan_volts,index):
            if index == 'XAxis':
                scan_x = float(scan_volts)
            if index == 'YAxis':
                scan_y = float(scan_volts)

        pixel_size, _, index = self._getPVStateShard(root,'micronsPerPixel')
        for pixel_size,index in zip(pixel_size,index):
            if index == 'XAxis':
                pix_sz_x = float(pixel_size)
            if index == 'YAxis':
                pix_sz_y = float(pixel_size)
        
        if n_planes == 1:
            n_frames = root.findall('Sequence/Frame')[-1].get('index') # use suite2p output instead later
        else: 
            n_frames = root.findall('Sequence')[-1].get('cycle')
    
        extra_params = root.find('Sequence/Frame/ExtraParameters')
        last_good_frame = extra_params.get('lastGoodFrame')

        self.fps = fps/n_planes
        self.frame_x = frame_x
        self.frame_y = frame_y
        self.n_planes = n_planes
        self.pix_sz_x = pix_sz_x
        self.pix_sz_y = pix_sz_y
        self.scan_x = scan_x
        self.scan_y = scan_y 
        self.zoom = zoom
        self.n_frames = int(n_frames)
        self.last_good_frame = last_good_frame

        print('n planes:', n_planes,
            '\nn frames:', int(n_frames),
            '\nlast good frame (0 = all good):', last_good_frame,
            '\nfps:', fps/n_planes,
            '\nframe size (px):', frame_x, 'x', frame_y, 
            '\nzoom:', zoom, 
            '\npixel size (um):', pix_sz_x, pix_sz_y,
            '\nscan centre (V):', scan_x, scan_y
            )

        
    def _parseNAPARMxml(self):
        '''
        From the NAPARM xml file, find relevant photostimulation metadata
        '''
        NAPARM_xml_path = path_finder(self.naparm_path, '.xml')[0];
        print('\nNAPARM xml:', NAPARM_xml_path)

        xml_tree = ET.parse(NAPARM_xml_path)
        root = xml_tree.getroot()

        title = root.get('Name')
        n_trials = int(root.get('Iterations'))

        for elem in root:
            if int(elem[0].get('InitialDelay')) > 0:
                inter_point_delay = int(elem[0].get('InitialDelay'))
                single_stim_dur = float(elem[0].get('Duration'))

        n_groups, n_reps, n_shots = [int(s) for s in re.findall(r'\d+', title)] 
        
        print('numbers of trials:', n_trials,
            '\ninter-group delay:', inter_point_delay,
            '\nnumber of groups:', n_groups,
            '\nnumber of shots:', n_shots,
            '\nnumber of sequence reps:', n_reps,
            '\nsingle stim duration (ms):', single_stim_dur
            )

        self.n_groups = n_groups
        self.n_reps = n_reps
        self.n_shots = n_shots
        self.inter_point_delay = inter_point_delay
        self.single_stim_dur = single_stim_dur

        
    def _parseNAPARMgpl(self):
        '''
        From the NAPARM gpl file, find relevant photostimulation metadata
        '''
        NAPARM_gpl_path = path_finder(self.naparm_path, '.gpl')[0];
        print('\nNAPARM gpl:', NAPARM_gpl_path)

        xml_tree = ET.parse(NAPARM_gpl_path)
        root = xml_tree.getroot()

        for elem in root:
            if elem.get('Duration'):
#                 single_stim_dur = float(elem.get('Duration'))
                spiral_size = float(elem.get('SpiralSize'))
                spiral_size = (spiral_size + 0.005155) / 0.005269
                break
        
#         print('single stim duration (ms):', single_stim_dur,
        print('\nspiral size (um):', int(spiral_size))

        self.spiral_size = int(spiral_size)
#         self.single_stim_dur = single_stim_dur

        
    def paqProcessing(self):
        '''
        Find the stimulation metadata from the .paq file including stim start frames,
        duration of stim (frames), frame clock and stim timings
        '''
        print('\nFinding stim frames from:', self.paq_path)
        
        paq_file = paq_read(self.paq_path)
        self.frame_clock = paq_data(paq_file, 'frame_clock', threshold_ttl=True, plot=False) 
        self.stim_times = paq_data(paq_file, self.stim_channel, threshold_ttl=True, plot=False)

        self.stim_start_frames = []
        
        for plane in range(self.n_planes):
            # this finds the nearest frame preceding the stim for each stim time
            stim_start_frames = stim_start_frame(paq_file, self.stim_channel, 
                                self.frame_clock, self.stim_times, plane, self.n_planes)

            self.stim_start_frames.append(stim_start_frames)
        
        # calculate number of frames corresponding to stim duration
        duration_ms = self.stim_dur
        frame_rate = self.fps
        duration_frames = np.ceil((duration_ms/1000)*frame_rate) + 1 # +1 for worst case scenarios
        self.duration_frames = int(duration_frames)
        print('total stim duration (frames):', int(duration_frames))
        
#         #sanity check - reduce stim_start_frames until it is below n_frames to catch missing frames
#         while np.amax(self.stim_start_frames) > self.n_frames:
#             for plane in range(self.n_planes):
#                 self.stim_start_frames[plane] = self.stim_start_frames[plane][:-1]


    def photostimProcessing(self):
        '''
        Find photostimulation metadata from the xml and gpl files in the NAPARM folder
        and calculate additional parameters from them
        '''
        self._parseNAPARMxml()
        self._parseNAPARMgpl()

        single_stim = self.single_stim_dur * self.n_shots
        total_single_stim = single_stim + self.inter_point_delay 
        
        total_multi_stim = total_single_stim * self.n_groups
        
        total_stim = total_multi_stim * self.n_reps
        
        self.stim_dur = total_stim
        self.stim_freq = (1000 / total_multi_stim) # stim frequency for one cell/group in Hz

        self.paqProcessing()

        
    def whiskerStimProcessing(self):
        '''
        Assign attributes for a whisker stim experiment
        '''
        self.stim_dur = 1000 # total whisker stim duration (ms)
        self.stim_freq = 10 # Hz
        
        self.paqProcessing()
        
        # calculate number of trials that could fit in the t-series
        # only really applicable for t-series with data loss at end of t-series
        trial_length = np.diff(self.stim_start_frames)
        trial_length_count = np.bincount(trial_length[0])
        mode_trial_length = np.argmax(trial_length_count)
        max_n_trials = (self.n_frames - self.stim_start_frames[0][0])/mode_trial_length
        
        if max_n_trials < 100:
            self.n_trials = int(math.floor(max_n_trials))
        else:
            self.n_trials = len(self.stim_start_frames[0])
            
        
    def stimProcessing(self):
        '''
        Depending on which stims were carried out during this session, process
        the data accordingly
        '''
        if self.stim_type == 'w':
            self.whiskerStimProcessing()
        if any(s in self.stim_type for s in ['pr', 'ps']):
            self.photostimProcessing()
    
    
    def _findS1S2(self, cell_med, s2_borders_path):
        '''
        Use coordinates of a bisecting line between S1 and S2 to determine which cells
        should be assigned as being in S1 or S2
        
        Inputs:
            cell_med        - the central coordinates of the cell
            s2_borders_path - folder containing .csv files with coordinates of a line
                              bisecting S1 and S2
        Output: 
            True/False      - whether the cell is in S1 or not
        '''
        y = cell_med[0]
        x = cell_med[1]

        for path in os.listdir(s2_borders_path):
            if all(s in path for s in ['.csv', self.sheet_name]):
                csv_path = os.path.join(s2_borders_path, path)

        xline = []
        yline = []

        with open(csv_path) as csv_file:
            csv_file = csv.DictReader(csv_file, fieldnames=None, dialect='excel')

            for row in csv_file:
                xline.append(int(row['xcoords']))
                yline.append(int(row['ycoords']))
        
        # sort the coordinates of the line and retrieve x and y coords
        # assumption that the line is monotonic
        line_argsort = np.argsort(yline)
        xline = np.array(xline)[line_argsort]
        yline = np.array(yline)[line_argsort]
        
        # determine which y-coordinates of the line the cell lies between
        i = bisect.bisect(yline, y)
        
        # if beyond the end of the line, put it between last two coords
        if i >= len(yline): 
            i = len(yline)-1
            
        # if before the start of the line, put it between the first two coords
        elif i == 0: 
            i = 1
        
        # determine which side of the line the cell lies (along x-axis)
        d = (x - xline[i])*(yline[i-1] - yline[i]) - (y-yline[i])*(xline[i-1]-xline[i])
        
        # make up two points at the far left and far right of the image
        # determine what the sign of the equation is for those two
#         frame_x = int(self.frame_x/2) 
        frame_x = self.frame_x # corrected this from above
        half_frame_y = int(self.frame_y/2)
        ds_left = (0-xline[i])*(yline[i-1]-yline[i]) - (half_frame_y-yline[i])*(xline[i-1]-xline[i])
        ds_right = (frame_x-xline[i])*(yline[i-1]-yline[i]) - (half_frame_y-yline[i])*(xline[i-1]-xline[i])

        # whichever sign the cell matches is the side the cell is on (left or right, S1 or S2)
        if np.sign(d) == np.sign(ds_left):
            return True
        elif np.sign(d) == np.sign(ds_right):
            return False
        else:
            return False
        
            
    def _retrieveS2pData(self, s2_borders_path):
        '''
        Extract suite2p data from stat.npy file and F.npy (using Fneu.npy for neuropil subtraction) 
        
        Inputs
            s2_borders_path - path to .csv files containing coordinates for bisecting line of S1/S2
        '''
        self.cell_id = [] # [cell]
        self.n_units = [] 
        self.cell_plane = [] # [cell]
        self.cell_med = [] # [cell[y,x]]
        self.cell_s1 = [] # Bool [cell]
        self.cell_s2 = [] # Bool [cell]
        self.num_s1_cells = []
        self.num_s2_cells = []
        self.cell_x = [] # [cell]
        self.cell_y = [] # [cell]
        self.raw = [] # [cell x time]
        self.dfof = [] # [cell x time]
        self.mean_img = [] 
        self.mean_imgE = []

        for plane in range(self.n_planes):
                
            # extract suite2p stat.npy and neuropil-subtracted F
            s2p_path = self.s2p_path
            FminusFneu, _, stat = s2p_loader(s2p_path, subtract_neuropil=True, neuropil_coeff=0.7)
            self.dfof.append(dfof2(FminusFneu[:,self.frames])) # calculate df/f based on relevant frames
            self.raw.append(FminusFneu[:,self.frames]) # raw F of each cell

            ops = np.load(os.path.join(s2p_path,'ops.npy'), allow_pickle=True).item()
            self.mean_img.append(ops['meanImg']) 
            self.mean_imgE.append(ops['meanImgE']) # enhanced mean image from suite2p
            self.xoff = ops['xoff'] # motion correction info
            self.yoff = ops['yoff']
                
            cell_id = []
            cell_plane = []
            cell_med = []
            cell_s1 = []
            cell_x = []
            cell_y = []
            
            # stat is an np array of dictionaries
            for cell,s in enumerate(stat): 
                cell_id.append(s['original_index']) 
                cell_med.append(s['med'])
                cell_s1.append(self._findS1S2(s['med'], s2_borders_path))
                
                cell_x.append(s['xpix'])
                cell_y.append(s['ypix'])
                
            self.cell_id.append(cell_id)
            self.n_units.append(len(self.cell_id[plane]))
            self.cell_med.append(cell_med)
            self.cell_s1.append(cell_s1)
            self.cell_s2.append(np.invert(cell_s1))
            self.num_s1_cells.append(np.sum(cell_s1))
            self.num_s2_cells.append(np.sum(np.invert(cell_s1)))
            self.cell_x.append(cell_x)
            self.cell_y.append(cell_y)

            num_units = FminusFneu.shape[0]
            cell_plane.extend([plane]*num_units)
            self.cell_plane.append(cell_plane)
    
    
    def _baselineFluTrial(self, flu_trial):
        '''
        Subtract baseline from dff trials to normalise across cells
        
        Inputs:
            flu_trial           - [cell x frame] dff trial for all cells
        Outputs:
            baselined_flu_trial - detrended dff trial with zeros replacing stim artifact
        '''        
        # baseline the flu_trial using pre-stim period mean flu for each cell
        baseline_flu = np.mean(flu_trial[:, :self.pre_frames], axis=1)
        # repeat the baseline_flu value across all frames for each cell
        baseline_flu_stack = np.repeat(baseline_flu, flu_trial.shape[1]).reshape(flu_trial.shape)
        # subtract baseline values for each cell
        baselined_flu_trial = flu_trial - baseline_flu_stack
        
        return baselined_flu_trial
    
    
    def _detrendFluTrial(self, flu_trial, stim_end):
        '''
        Detrend dff trials to account for drift of signal over a trial
        
        Inputs:
            flu_trial           - [cell x frame] dff trial for all cells
            stim_end            - frame n of the stim end
        Outputs:
            detrended_flu_trial - detrended dff trial with zeros replacing stim artifact
        '''        
        # remove stim artifact from the trial (truncates the array)
        no_stim_flu_trial = np.delete(flu_trial, range(self.pre_frames, stim_end), axis=1) # CHANGE THIS TO stim_end + 1?
        
        # detrend and baseline-subtract the flu trial for all cells
        detrended_flu_trial = signal.detrend(no_stim_flu_trial, axis=1)
        baselined_flu_trial = self._baselineFluTrial(detrended_flu_trial)
        
        # insert zeros in place of stim artifact
        processed_flu_trial = np.insert(baselined_flu_trial, [self.pre_frames]*self.duration_frames, 0, axis=1)
                
        return processed_flu_trial
    
    
    def _makeFluTrials(self, plane_flu, plane):
        '''
        Take dff trace and for each trial, correct for drift in the recording and baseline subtract
        
        Inputs:
            plane_flu   - dff traces for all cells for this plane only
            plane       - imaging plane n
        Outputs:
            trial_array - detrended, baseline-subtracted trial array [cell x frame x trial]
        '''
        
        print('finding trials')
        
        for i, stim in enumerate(self.stim_start_frames[plane]):
            # get frame indices of entire trial from pre-stim start to post-stim end
            trial_frames = np.s_[stim - self.pre_frames : stim + self.post_frames]
            
            # use trial frames to extract this trial for every cell
            flu_trial = plane_flu[:,trial_frames]
            
            stim_end = self.pre_frames + self.duration_frames
            
            # catch timeseries which ended too early
            if flu_trial.shape[1] > stim_end:
                # don't detrend whisker stim data
                if any(s in self.stim_type for s in ['pr', 'ps', 'none']):
                    # detrend only the flu_trial outside of stim artifact and baseline
#                     flu_trial = self._detrendFluTrial(flu_trial, stim_end)
                    flu_trial = self._baselineFluTrial(flu_trial)
                else:
                    flu_trial = self._baselineFluTrial(flu_trial)

            # only append trials of the correct length - will catch corrupt/incomplete data and not include
            if i == 0:
                trial_array = flu_trial
                flu_trial_shape = flu_trial.shape[1]
            else:
                if flu_trial.shape[1] == flu_trial_shape:
                    trial_array = np.dstack((trial_array, flu_trial))
                else:
                    print('**incomplete trial detected and not appended to trial_array**', end='\r')
                    
        return trial_array
    
    
    def _trialProcessing(self, plane):
        '''
        Take dfof trace for entire timeseries and break it up in to individual trials, calculate
        the mean amplitudes of response and statistical significance across all trials
        
        Inputs:
            plane             - imaging plane n
        '''
        # make trial arrays from dff data [plane x cell x frame x trial]
        trial_array = self._makeFluTrials(self.dfof[plane], plane)
        self.all_trials.append(trial_array)
        self.n_trials = trial_array.shape[2]
        
        # mean pre and post stimulus flu values for all cells, all trials
        pre_array = np.mean(trial_array[:, self.pre_trial_frames, :], axis=1)
        post_array = np.mean(trial_array[:, self.post_trial_frames, :], axis=1)
        
        # calculate amplitude of response for all cells, all trials
        all_amplitudes = post_array - pre_array
        self.all_amplitudes.append(all_amplitudes)
        
        # check if the two distributions of flu values (pre/post) are different
#         t_tests = stats.ttest_rel(pre_array, post_array, axis=1)
#         self.t_tests.append(t_tests[1][:])
        wilcoxons = np.empty(self.n_units[0]) # [cell (p-value)]
        
        for cell in range(self.n_units[0]):
            wilcoxons[cell] = stats.wilcoxon(post_array[cell], pre_array[cell])[1]

        self.wilcoxons.append(wilcoxons)
    
    
    def _STAProcessing(self, plane):
        '''
        Make stimulus triggered average (STA) traces and calculate the STA amplitude
        of response
        
        Input:
            plane - imaging plane n
        '''
        # make stas, [plane x cell x frame]
        stas = np.mean(self.all_trials[plane], axis=2)
        self.stas.append(stas)

        # make sta amplitudes, [plane x cell]
        pre_sta = np.mean(stas[:, self.pre_trial_frames], axis=1)
        post_sta = np.mean(stas[:, self.post_trial_frames], axis=1)
        sta_amplitudes = post_sta - pre_sta
        self.sta_amplitudes.append(sta_amplitudes)
    
    
    def _sigTestTrialDFSF(self, plane):
        '''
        Calculate change in fluorescence, normalise using the standard deviation of the baseline (dfsf)
        and determine if cell responded (either negative or positive) on each trial (>1 std of baseline)
        
        Input:
            plane - imaging plane n
        '''
        # make dF/sF trials to calculate single trial responses from
        df_trials = self._makeFluTrials(self.raw[plane], plane)   
        std_baseline = np.std(df_trials[:, :self.pre_frames, :], axis=1) # std of baseline period
        std_baseline = np.expand_dims(std_baseline, axis=1)
        std_baseline = np.repeat(std_baseline, df_trials.shape[1], axis=1)
        dfsf_trials = df_trials/std_baseline

        mean_post_dfsf = np.nanmean(dfsf_trials[:, self.post_trial_frames, :], axis=1)

        trial_sig = np.absolute(mean_post_dfsf > 1)
        
        self.trial_sig_dfsf.append(trial_sig)
        
    
    def _sigTestTrialDFF(self, plane):
        '''
        Use detrended, baseline-subtracted dF/F trials to determine if the cell responded
        on each trial in the post-stim period at >1 std of baseline
        
        Input:
            plane - imaging plane n
        '''
        dff_baseline = self.all_trials[plane][:, :self.pre_frames, :] # [cell x frames x trial]
        std_baseline = np.std(dff_baseline, axis=1)

        trial_sig = np.absolute(self.all_amplitudes[plane]) >= 1*std_baseline # [cell x trial]
            
        # [plane x cell x trial]
        self.trial_sig_dff.append(trial_sig)
        
        
    def _sigTestAvgDFF(self, plane):
        '''
        Uses the p values and a threshold for the Benjamini-Hochberg correction to return which 
        cells are still significant after correcting for multiple significance testing
        '''
              
#         p_vals = self.t_tests[plane]
        p_vals = self.wilcoxons[plane]

        # multiple comparison correction using benjamini hochberg ('fdr_bh')
        #         sig_units, _, _, _ = smstats.multitest.multipletests(p_vals, alpha=0.1, method='fdr_bh', 
        #                                                              is_sorted=False, returnsorted=False)

        s1_cells = np.array(self.cell_s1[plane])
        s2_cells = np.array(self.cell_s2[plane])

        if any(s in self.stim_type for s in ['pr', 'ps', 'none']):
            targets = self.targeted_cells
            s1_p = p_vals[~targets&s1_cells]
            s2_p = p_vals[~targets&s2_cells]
            target_p = p_vals[targets]
        else:
            s1_p = p_vals[s1_cells]
            s2_p = p_vals[s2_cells]

        sig_units = np.full_like(p_vals, False, dtype=bool)

        alpha = 0.1 
        
        try:
            s1_sig, _, _, _ = smstats.multitest.multipletests(s1_p, alpha=alpha, method='fdr_bh', 
                                                                     is_sorted=False, returnsorted=False)
        except ZeroDivisionError:
            print('no S1 cells responding')
            s1_sig = sig_units[~targets&s1_cells]
            
        try:
            s2_sig, _, _, _ = smstats.multitest.multipletests(s2_p, alpha=alpha, method='fdr_bh', 
                                                                         is_sorted=False, returnsorted=False)
        except ZeroDivisionError:
            print('no S2 cells responding')
            s2_sig = sig_units[~targets&s2_cells]
            
        if any(s in self.stim_type for s in ['pr', 'ps', 'none']):
            target_sig, _, _, _ = smstats.multitest.multipletests(target_p, alpha=alpha, method='fdr_bh', 
                                                                         is_sorted=False, returnsorted=False)
            sig_units[targets] = target_sig
            sig_units[~targets&s1_cells] = s1_sig
            sig_units[~targets&s2_cells] = s2_sig
        else:
            sig_units[s1_cells] = s1_sig
            sig_units[s2_cells] = s2_sig

        # p values without bonferroni correction
        no_bonf_corr = [i for i,p in enumerate(p_vals) if p < 0.05]
        nomulti_sig_units = np.zeros(self.n_units[plane], dtype='bool')
        nomulti_sig_units[no_bonf_corr] = True
        
        # p values after bonferroni correction
#         bonf_corr = [i for i,p in enumerate(p_vals) if p < 0.05 / self.n_units[plane]]
#         sig_units = np.zeros(self.n_units[plane], dtype='bool')
#         sig_units[bonf_corr] = True

        self.sta_sig.append(sig_units)  
        self.sta_sig_nomulti.append(nomulti_sig_units)

        
    def _probResponse(self, plane, trial_sig_calc):
        '''
        Calculate the response probability, i.e. proportion of trials that each cell responded on
        
        Inputs:
            plane          - imaging plane n
            trial_sig_calc - indicating which calculation was used for significance testing ('dff'/'dfsf')
        '''
        n_trials = self.n_trials

        # get the number of responses for each across all trials
        if trial_sig_calc == 'dff':
            num_respond = np.array(self.trial_sig_dff[plane]) # trial_sig_dff is [plane][cell][trial]
        elif trial_sig_calc == 'dfsf': 
            num_respond = np.array(self.trial_sig_dfsf[plane])
        
        # calculate the proportion of all trials that each cell responded on
        self.prob_response.append(np.sum(num_respond, axis=1) / n_trials)
    
    
    def _makeTimeVector(self):
        '''
        Create vector of frame times in milliseconds rather than frames
        '''
        n_frames = self.all_trials[0].shape[1]
        pre_time = self.pre_frames/self.fps
        post_time = self.post_frames/self.fps
        self.time = np.linspace(-pre_time, post_time, n_frames)
        
    
    def _cropTargets(self, target_image_scaled):
        '''
        Crop the target image based on scan field centre and FOV size in pixels
        
        Inputs:
            target_image_scaled - 8-bit image with single pixels of 255 indicating target locations
        '''                  
        # make a bounding box centred on (512,512)
        x1 = 511 - self.frame_x/2
        x2 = 511 + self.frame_x/2
        y1 = 511 - self.frame_y/2
        y2 = 511 + self.frame_y/2
        
        # scan amplitudes in voltage from PrairieView software for 1x FOV
        ScanAmp_X = 2.62*2
        ScanAmp_Y = 2.84*2
        
        # scale scan amplitudes according to the imaging zoom
        ScanAmp_V_FOV_X = ScanAmp_X / self.zoom
        ScanAmp_V_FOV_Y = ScanAmp_Y / self.zoom
        
        # find the voltage per pixel
        scan_pix_y = ScanAmp_V_FOV_Y / 1024
        scan_pix_x = ScanAmp_V_FOV_X / 1024

        # find the offset (if any) of the scan field centre, in pixels
        offset_x = self.scan_x/scan_pix_x
        offset_y = self.scan_y/scan_pix_y

        # offset the bounding box by the appropriate number of pixels
        x1,x2,y1,y2 = round(x1+offset_x), round(x2+offset_x), round(y1-offset_y), round(y2-offset_y)

        # crop the target image using the offset bounding box to put the targets in imaging space
        return target_image_scaled[y1:y2, x1:x2]
        
                          
    def _findTargetAreas(self):
        '''
        Use NAPARM target file to find target locations and expand their size to create target masks
        '''                  
        print('\nFinding SLM target locations...')
        
        # load naparm targets file for this experiment
        naparm_path = os.path.join(self.naparm_path, 'Targets')

        listdir = os.listdir(naparm_path)

        for path in listdir:
            if 'AllFOVTargets' in path:
                target_file = path
        
        target_image = tf.imread(os.path.join(naparm_path, target_file))
        
        # target image is 512x512, so scale to 1024x1024 (usual image size)
        n = np.array([[0, 0],[0, 1]], dtype='uint8')
        target_image_scaled = np.kron(target_image, n)
        
        # crop the image if smaller than 1024x1024
        if self.frame_x < 1024 or self.frame_y < 1024:
            target_image_scaled = self._cropTargets(target_image_scaled)
            tf.imwrite(os.path.join(naparm_path, 'target_image_scaled.tif'), target_image_scaled)
        else:
            tf.imwrite(os.path.join(naparm_path, 'target_image_scaled.tif'), target_image_scaled)
        
        # find target coordinates, which are 255 on background of 0
        targets = np.where(target_image_scaled>0)
        
        targ_coords = list(zip(targets[0], targets[1]))
        print('number of targets:', len(targ_coords))

        self.target_coords = targ_coords
        self.n_targets = len(targ_coords)
        
        # make target areas by expanding each point to the size of the spiral (normally 10 um) + 10 um
        target_areas = []

        radius = int(((self.spiral_size/2)+10)/self.pix_sz_x) # adding 10 um for photostim res
        for coord in targ_coords:
            target_area = np.array([item for item in points_in_circle_np(radius, x0=coord[0], y0=coord[1])])
            if not any([max(target_area[:,1]) > self.frame_x,
                        max(target_area[:,0]) > self.frame_y,
                        min(target_area[:,1]) < 0,
                        min(target_area[:,0]) < 0
                       ]):
                target_areas.append(target_area)
            else:
                print('\n **** TARGET AREA OUTSIDE OF FRAME ****')

        self.target_areas = target_areas
    
                          
    def _findTargetedCells(self):
        '''
        Make a binary mask of the targets and multiply by an image of the cells 
        to find cells that were targeted
        '''
        print('searching for targeted cells...')
        
        # make all target area coords in to a binary mask
        targ_img = np.zeros([self.frame_x, self.frame_y], dtype='uint16')
        target_areas = np.array(self.target_areas)
        targ_img[target_areas[:,:,1], target_areas[:,:,0]] = 1
        
        # make an image of every cell area, filled with the index of that cell
        cell_img = np.zeros_like(targ_img)
        
        cell_x = np.array(self.cell_x)
        cell_y = np.array(self.cell_y)

        for i,coord in enumerate(zip(cell_x[0], cell_y[0])):
            cell_img[coord] = i+1
        
        # binary mask x cell image to get the cells that overlap with target areas
        targ_cell = cell_img*targ_img
        
        targ_cell_ids = np.unique(targ_cell)[1:]-1 # correct the cell id due to zero indexing
        self.targeted_cells = np.zeros([self.n_units[0]], dtype='bool')
        self.targeted_cells[targ_cell_ids] = True 

        self.n_targeted_cells = np.sum(self.targeted_cells)

        print('search completed.')
        print('Number of targeted cells: ', self.n_targeted_cells)
            
            
    def _targetSumDff(self):
        '''
        Find the sum of dff response amplitudes for all responsive targets on each trial
        '''
        trial_amplitudes = self.all_amplitudes[0]
        
        # for each trial find targeted cells that responded
        trial_responders = self.trial_sig_dff[0]
        targeted_cells = np.repeat(self.targeted_cells[..., None], 
                                   trial_responders.shape[1], 1) # [..., None] is a quick way to expand_dims
        targeted_responders = targeted_cells & trial_responders
        
        # get only the target cell amplitudes and sum for each trial
        trial_target_amplitudes = np.multiply(trial_amplitudes, targeted_responders)
        trial_target_dff = np.sum(trial_target_amplitudes, axis=0)
        
        return trial_target_dff
    
    
    def _euclidDist(self, resp_positions):
        '''
        Find the mean Euclidean distance of all cells from a central point
        as a measure of spread
        
        Inputs:
            resp_positions - the median coordinates of cells
        Outputs:
            euclid_dist    - the mean Euclidean distance from central point
        '''
        # find centroid of all responding cells
        resp_coords = list(zip(*resp_positions))
        centroidx = np.mean(resp_coords[0])
        centroidy = np.mean(resp_coords[1])
        centroid = [centroidx, centroidy]
        
        # find distances of each cell from centroid
        dists = np.empty(resp_positions.shape[0])

        for i, resp in enumerate(resp_positions):
            dists[i] = np.linalg.norm(resp - centroid)
        
        euclid_dist = np.mean(dists) # take average as a measure of spread

        return euclid_dist

    
    def _targetSpread(self):
        '''
        Find the mean Euclidean distance of responding targeted cells (trial-wise and trial average)
        '''
        # for each trial find targeted cells that responded
        trial_responders = self.trial_sig_dff[0]
        targeted_cells = np.repeat(self.targeted_cells[..., None], 
                                   trial_responders.shape[1], 1) # [..., None] is a quick way to expand_dims
        targeted_responders = targeted_cells & trial_responders
        
        cell_positions = np.array(self.cell_med[0])
        
        dists = np.empty(self.n_trials)
        
        # for each trial, find the spread of responding targeted cells
        for i, trial in enumerate(range(self.n_trials)):
            resp_cell = np.where(targeted_responders[:,trial])
            resp_positions = cell_positions[resp_cell]

            if resp_positions.shape[0] > 1: # need more than 1 cell to measure spread...
                dists[i] = self._euclidDist(resp_positions)
            else:
                dists[i] = np.nan

        self.trial_euclid_dist = dists

        # find spread of targets that statistically significantly responded over all trials
        responder = self.sta_sig[0]
        targeted_responders = responder & self.targeted_cells

        resp_cell = np.where(targeted_responders)
        resp_positions = cell_positions[resp_cell]

        if resp_positions.shape[0] > 1: # need more than 1 cell to measure spread...
            dist = self._euclidDist(resp_positions)
        else: 
            dist = np.nan

        self.sta_euclid_dist = dist
    
    
    def _targetAnalysis(self):
        '''
        Find targets, assign cells as targeted, calculate total target input activity
        and target spread (euclidean distance from central point)
        '''
        self._findTargetAreas()
        
        self._findTargetedCells()

        self.trial_target_dff = self._targetSumDff()

                          
    def s2pAnalysis(self, s2_borders_path, trial_sig_calc='dff'):
        '''
        Take Suite2p outputs, collate relevant metadata, process raw data and analyse metrics
        on individual trials as well as across trial averages
        
        Inputs:
            s2_borders_path - path to .csv files containing coordinates for bisecting line of S1/S2
            trial_sig_calc  - 'dff' or 'dfsf' method used to calculate significant trial responses
        '''
        if self.n_frames == 0:
            print('______________________________________________________________________')
            print('\nNo s2p data for', self.stim_type, 'in this session')
            print('______________________________________________________________________')

        else: 
            print('______________________________________________________________________')
            print('\nProcessing s2p data for', self.stim_type, 'experiment type')
            print('______________________________________________________________________\n')
            
            # collect data from s2p
            self._retrieveS2pData(s2_borders_path)
            
            # process s2p data
            if self.stim_start_frames:
    
                # pre-allocated empty lists
                self.all_trials = [] # all trials for each cell, dff detrended
                self.all_amplitudes = [] # all amplitudes of response between dff test periods
                
                self.stas = [] # avg of all trials for each cell, dff
                self.sta_amplitudes = [] # avg amplitude of response between dff test periods
                
                self.prob_response = [] # proportion of trials responding on
                self.t_tests = [] # result from related samples t-test between dff test periods
                self.wilcoxons = []
                self.trial_sig_dff = [] # based on dff increase above std of baseline
                self.trial_sig_dfsf = [] # based on df/std(f) increase in test period post-stim
                self.sta_sig = [] # based on t-test between dff test periods
                self.sta_sig_nomulti = [] # as above, no multiple comparisons correction

                self.pre_frames = int(np.ceil(self.fps*2)) # pre-stim period to include in trial
                self.post_frames = int(np.ceil(self.fps*10)) # post-stim period to include in trial
                self.test_frames = int(np.ceil(self.fps*0.5)) # test period for stats

                for plane in range(self.n_planes):
                    
                    self.pre_trial_frames = np.s_[self.pre_frames - self.test_frames : self.pre_frames]
                    stim_end = self.pre_frames + self.duration_frames
                    self.post_trial_frames = np.s_[stim_end : stim_end + self.test_frames]
        
                    self._trialProcessing(plane)

                    self._STAProcessing(plane)
                    
                    self._sigTestTrialDFSF(plane)
                    self._sigTestTrialDFF(plane)
                    
                    # target cells
                    if any(s in self.stim_type for s in ['pr', 'ps', 'none']):
                        self._targetAnalysis()
                    
                    self._sigTestAvgDFF(plane)
                    
                    if any(s in self.stim_type for s in ['pr', 'ps', 'none']):
                        self._targetSpread()
                        
                    self._probResponse(plane, trial_sig_calc)
                
                self._makeTimeVector()          
                