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
import csv
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

        self.ss_id = ss_id
        self.sheet_name = sheet_name
        print('\n================================================================')
        print('Fetching paths and stim types for:', sheet_name)
        print('================================================================')
        
        info = self._getExperimentMetadata(ss_id, sheet_name, pstation_path)

        self.s2p_path = os.path.join(info[0][0], 'suite2p', 'plane0')

        self.photostim_r = interarealAnalysis(info[0], 'markpoints2packio', 'pr', self.s2p_path, self.sheet_name)
        self.photostim_s = interarealAnalysis(info[1], 'markpoints2packio', 'ps', self.s2p_path, self.sheet_name)
        self.whisker_stim = interarealAnalysis(info[2], 'piezo_stim', 'w', self.s2p_path, self.sheet_name)
        self.spont = interarealAnalysis(info[3], None, 'none', self.s2p_path, self.sheet_name)

        
    def _sortPaths(self, tiffs_pstation, naparms_pstation, paqs_pstation, stim_type):

        pr_id = np.where(stim_type=='pr') #photostim random groups
        ps_id = np.where(stim_type=='ps') #photostim similar groups
        w_id = np.where(stim_type=='w') #whisker stim
        n_id = np.where(stim_type=='n') #no stim (spontaneous)
        
        photostim_similar = np.concatenate((tiffs_pstation[ps_id], naparms_pstation[ps_id], paqs_pstation[ps_id]))
        photostim_random = np.concatenate((tiffs_pstation[pr_id], naparms_pstation[pr_id], paqs_pstation[pr_id]))
        whisker_stim = np.concatenate((tiffs_pstation[w_id], naparms_pstation[w_id], paqs_pstation[w_id]))
        spontaneous = np.concatenate((tiffs_pstation[n_id], naparms_pstation[n_id], paqs_pstation[n_id]))

        return photostim_random, photostim_similar, whisker_stim, spontaneous

    
    def _getExperimentMetadata(self, ss_id, sheet_name, pstation_path):
        
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
            if stim=='pr': stim_type.append('pr')
            if stim=='ps': stim_type.append('ps')
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
        print('\nUmbrella folder:', umbrella)

        for i in for_processing.index:
            tiff_path = for_processing.loc[i, 'tiff_path'].replace('"', '')
            tiff_path = ntpath.basename(tiff_path)
            tiff_path = path_finder(umbrella, tiff_path, is_folder=True)
            tiffs_pstation.append(tiff_path[0]) # convert paths (from Packer1 or PackerStation) to local QNAP paths
            
            naparm_path = for_processing.loc[i, 'naparm_path'].replace('"', '')
            naparm_path = ntpath.basename(naparm_path)
            if naparm_path:
                naparm_path = path_finder(umbrella, naparm_path, is_folder=True)
            else:
                naparm_path = ['none']
            naparms_pstation.append(naparm_path[0]) # convert paths (from Packer1 or PackerStation) to local QNAP paths
            
            paq_path = for_processing.loc[i, 'paq_path'].replace('"', '')
            paq_path = ntpath.basename(paq_path)
            if paq_path:
                paq_path = path_finder(umbrella, paq_path, is_folder=False)
            else:
                paq_path = ['none']
            paqs_pstation.append(paq_path[0]) # convert paths (from Packer1 or PackerStation) to local QNAP paths
        
        tiffs_pstation = np.array(tiffs_pstation)

        naparms_pstation = np.array(naparms_pstation)

        paqs_pstation = np.array(paqs_pstation)

        stim_type = np.array(stim_type)

        info = self._sortPaths(tiffs_pstation, naparms_pstation, paqs_pstation, stim_type)
        print('\nExperimental info =', info)

        return info
    
    
    def _getPhotostimFrames(self, obj):
        
        photostim_frames = []

        stim_duration_frames = list(range(0,obj.duration_frames))

        for frame in obj.stim_start_frames[0]:
            new_frames = stim_duration_frames + frame
            photostim_frames.extend(new_frames)
            
        return photostim_frames
    
    
    def s2pRun(self):

        sampling_rate = self.photostim_r.fps/self.photostim_r.n_planes
        diameter_x = 13/self.photostim_r.pix_sz_x
        diameter_y = 13/self.photostim_r.pix_sz_y
        diameter = int(diameter_x), int(diameter_y)
        n_planes = self.photostim_r.n_planes

        data_folders = [
                        self.photostim_r.tiff_path, 
                        self.photostim_s.tiff_path,
                        ]

        if self.whisker_stim.tiff_path:
            data_folders.extend([self.whisker_stim.tiff_path])

        if self.spont.tiff_path:
            data_folders.extend([self.spont.tiff_path])

        # find the photostim frames from photostim t-series
        photostim_r_frames = np.array(self._getPhotostimFrames(self.photostim_r))
        photostim_s_frames = np.array(self._getPhotostimFrames(self.photostim_s)) + self.photostim_r.n_frames
        photostim_frames = np.concatenate((photostim_r_frames, photostim_s_frames))
        first_data_folder = self.photostim_r.tiff_path
        np.save(os.path.join(first_data_folder, 'bad_frames.npy'), photostim_frames) 

        db = {
            'data_path' : data_folders, 
            'fs' : float(sampling_rate),
            'diameter' : diameter, 
            'nplanes' : n_planes
            }

        print('\ns2p ops:', db)

        run_s2p(ops=ops,db=db)
    
    
    def getFrameRanges(self):

        s2p_path = self.s2p_path
        ops_path = os.path.join(s2p_path, 'ops.npy')
        ops = np.load(ops_path, allow_pickle=True)
        ops = ops.item()

        frame_list = ops['frames_per_folder']
        self.frame_list = frame_list

        i = 0

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

        print('\nCalculating frame ranges...')
        self.photostim_r.frames = range(0,self.photostim_r.n_frames)
        subtotal = self.photostim_r.n_frames
        self.photostim_s.frames = range(subtotal,self.photostim_s.n_frames+subtotal)
        subtotal += self.photostim_s.n_frames
        self.whisker_stim.frames = range(subtotal,self.whisker_stim.n_frames+subtotal)
        subtotal += self.whisker_stim.n_frames
        self.spont.frames = range(subtotal,self.spont.n_frames+subtotal)
        
        
    def addShamPhotostim(self):
        
        self.spont.stim_start_frames = self.photostim_r.stim_start_frames
        self.spont.naparm_path = self.photostim_r.naparm_path
        self.spont.spiral_size = self.photostim_r.spiral_size 
        self.spont.duration_frames = self.photostim_r.duration_frames
        self.spont.stim_dur = self.photostim_r.stim_dur
        self.spont.stim_freq = self.photostim_r.stim_freq
        self.spont.single_stim_dur = self.photostim_r.single_stim_dur
        self.spont.n_shots = self.photostim_r.n_shots
        self.spont.n_groups = self.photostim_r.n_groups
        self.spont.n_trials = self.photostim_r.n_trials
        self.spont.inter_point_delay = self.photostim_r.inter_point_delay
        self.spont.targeted_cells = self.photostim_r.targeted_cells
        self.spont.n_targets = self.photostim_r.n_targets

        
    def _targetedWhiskerCells(self, exp_obj):
        
        trial_responders = exp_obj.trial_sig_dff[0]

        targeted_cells = np.repeat(exp_obj.targeted_cells[..., None], 
                                   trial_responders.shape[1], 1) # [..., None] is a quick way to expand_dims

        whisker_cells = np.repeat(self.whisker_stim.sta_sig[0][..., None],
                                  trial_responders.shape[1], 1)

        targeted_w_responders = targeted_cells & whisker_cells & trial_responders

        return targeted_w_responders
    
    
    def whiskerTargets(self):
        
        if self.whisker_stim.n_frames > 0:
            self.photostim_r.trial_w_targets = self._targetedWhiskerCells(self.photostim_r)
            self.photostim_s.trial_w_targets = self._targetedWhiskerCells(self.photostim_s)
            self.spont.trial_w_targets = self._targetedWhiskerCells(self.spont)
    
    
        
class interarealAnalysis():

    
    def __init__(self, info, stim_channel, stim_type, s2p_path, sheet_name):
        
        if any(info):            
            self.tiff_path = info[0]
            self.naparm_path = info[1]
            self.paq_path = info[2]
            
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
    
    
    def _getPVStateShard(self, path, key):

        value = []
        description = []
        index = []

        tree = ET.parse(path) # parse xml from a path
        root = tree.getroot() # make xml tree structure

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
                for file in os.listdir(tiff_path):
                    if file.endswith('.xml'):
                        path = os.path.join(tiff_path, file)
                    if file.endswith('.env'):
                        env_path = os.path.join(tiff_path, file) 
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

        zoom = float(self._getPVStateShard(path,'opticalZoom')[0])

        scanVolts, _, index = self._getPVStateShard(path,'currentScanCenter')
        for scanVolts,index in zip(scanVolts,index):
            if index == 'XAxis':
                scan_x = float(scanVolts)
            if index == 'YAxis':
                scan_y = float(scanVolts)

        pixelSize, _, index = self._getPVStateShard(path,'micronsPerPixel')
        for pixelSize,index in zip(pixelSize,index):
            if index == 'XAxis':
                pix_sz_x = float(pixelSize)
            if index == 'YAxis':
                pix_sz_y = float(pixelSize)
        
        if n_planes == 1:
            n_frames = root.findall('Sequence/Frame')[-1].get('index') # not useful if xml writing breaks, use suite2p output instead
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

        NAPARM_xml_path = path_finder(self.naparm_path, '.xml')[0]
        print('\nNAPARM xml:', NAPARM_xml_path)

        xml_tree = ET.parse(NAPARM_xml_path)
        root = xml_tree.getroot()

        title = root.get('Name')
        n_trials = int(root.get('Iterations'))

        for elem in root:
            if int(elem[0].get('InitialDelay')) > 0:
                inter_point_delay = int(elem[0].get('InitialDelay'))

        n_groups, n_reps, n_shots = [int(s) for s in re.findall(r'\d+', title)]

        print('numbers of trials:', n_trials,
            '\ninter-group delay:', inter_point_delay,
            '\nnumber of groups:', n_groups,
            '\nnumber of shots:', n_shots,
            '\nnumber of sequence reps:', n_reps,
            )

        self.n_groups = n_groups
        self.n_reps = n_reps
        self.n_shots = n_shots
        self.inter_point_delay = inter_point_delay

        
    def _parseNAPARMgpl(self):

        NAPARM_gpl_path = path_finder(self.naparm_path, '.gpl')[0]
        print('\nNAPARM gpl:', NAPARM_gpl_path)

        xml_tree = ET.parse(NAPARM_gpl_path)
        root = xml_tree.getroot()

        for elem in root:
            if elem.get('Duration'):
                single_stim_dur = float(elem.get('Duration'))
                spiral_size = float(elem.get('SpiralSize'))
                spiral_size = (spiral_size + 0.005155) / 0.005269
                break
        
        print('single stim duration (ms):', single_stim_dur,
            '\nspiral size (um):', int(spiral_size))

        self.spiral_size = int(spiral_size)
        self.single_stim_dur = single_stim_dur

        
    def paqProcessing(self):

        print('\nFinding stim frames from:', self.paq_path)
        
        paq = paq_read(self.paq_path)
        self.frame_clock = paq_data(paq, 'frame_clock', threshold_ttl=True, plot=False)
        self.stim_times = paq_data(paq, self.stim_channel, threshold_ttl=True, plot=False)

        self.stim_start_frames = []

        for plane in range(self.n_planes):

            stim_start_frames = stim_start_frame(paq, self.stim_channel, 
                                self.frame_clock, self.stim_times, plane, self.n_planes)

            self.stim_start_frames.append(stim_start_frames)
        
        # calculate number of frames corresponding to stim period
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

        self._parseNAPARMxml()
        self._parseNAPARMgpl()

        single_stim = self.single_stim_dur * self.n_shots
        total_single_stim = single_stim + self.inter_point_delay 
        
        total_multi_stim = total_single_stim * self.n_groups
        
        total_stim = total_multi_stim * self.n_reps
        
        self.stim_dur = total_stim - self.inter_point_delay
        self.stim_freq = (1000 / total_multi_stim) # stim frequency for one cell/group in Hz

        self.paqProcessing()

        
    def whiskerStimProcessing(self):

        self.stim_dur = 1000 # total whisker stim duration
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

        if self.stim_type == 'w':
            self.whiskerStimProcessing()
        if any(s in self.stim_type for s in ['pr', 'ps']):
            self.photostimProcessing()
    
    
    def _findS1S2(self, cell_med, s2_borders_path):
    
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
        
        #assumption = line is monotonic
        line_argsort = np.argsort(yline)
        xline = np.array(xline)[line_argsort]
        yline = np.array(yline)[line_argsort]
        
        i = bisect.bisect(yline, y)
        if i >= len(yline) : i = len(yline)-1
        elif i == 0 : i = 1
        
        frame_x = int(self.frame_x/2)
        half_frame_y = int(self.frame_y/2)

        d = (x - xline[i])*(yline[i-1] - yline[i]) - (y-yline[i])*(xline[i-1]-xline[i])

        ds1 = (0 - xline[i])*(yline[i-1] - yline[i]) - (half_frame_y-yline[i])*(xline[i-1]-xline[i])
        ds2 = (frame_x - xline[i])*(yline[i-1] - yline[i]) - (half_frame_y-yline[i])*(xline[i-1]-xline[i])

        if np.sign(d) == np.sign(ds1):
            return True
        elif np.sign(d) == np.sign(ds2):
            return False
        else:
            return False
        
            
    def _retrieveS2pData(self, s2_borders_path):
        
        self.cell_id = []
        self.n_units = []
        self.cell_plane = []
        self.cell_med = []
        self.cell_s1 = []
        self.cell_s2 = []
        self.num_s1_cells = []
        self.num_s2_cells = []
        self.cell_x = []
        self.cell_y = []
        self.raw = []
        self.dfof = []
        self.mean_img = []
        self.mean_imgE = []

        for plane in range(self.n_planes):
                
            s2p_path = self.s2p_path
            FminusFneu, _, stat = s2p_loader(s2p_path, subtract_neuropil=True, neuropil_coeff=0.7)
            dff = dfof2(FminusFneu)
            self.raw.append(FminusFneu[:,self.frames])
            self.dfof.append(dff[:,self.frames])

            ops = np.load(os.path.join(s2p_path,'ops.npy'), allow_pickle=True).item()
            self.mean_img.append(ops['meanImg'])
            self.mean_imgE.append(ops['meanImgE'])
            self.xoff = ops['xoff']
            self.yoff = ops['yoff']
                
            cell_id = []
            cell_plane = []
            cell_med = []
            cell_s1 = []
            cell_x = []
            cell_y = []
                
            for cell,s in enumerate(stat):
                cell_id.append(s['original_index']) # stat is an np array of dictionaries!
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
    
    
    def _detrendFluTrial(self, flu_trial, stim_end):
        
        flu_trial = np.delete(flu_trial, range(self.pre_frames, stim_end), axis=1)
        flu_trial = signal.detrend(flu_trial, axis=1, overwrite_data=True)
        flu_trial = np.insert(flu_trial, [self.pre_frames]*self.duration_frames, 0, axis=1)
                
        return flu_trial
    
    
    def _makeFluTrials(self, plane_flu, plane):
        
        # make detrended, baseline-subtracted flu trial arrays of shape [cell x frame x trial]
        for i, stim in enumerate(self.stim_start_frames[plane]):
            # use stim start frames and get before and after period based on pre_ or post_frames
            trial_frames = np.s_[stim - self.pre_frames : stim + self.post_frames]

            # make flu_trial from plane_dff
            flu_trial = plane_flu[:,trial_frames]
            
            stim_end = self.pre_frames + self.duration_frames
            
            # detrend the dfof trace
            if flu_trial.shape[1] > stim_end:
                
                if any(s in self.stim_type for s in ['pr', 'ps', 'none']):
                    # detrend only the flu_trial outside of stim artifact
                    flu_trial = self._detrendFluTrial(flu_trial, stim_end)
                    
                # baseline flu_trial to first 2 seconds
                baseline_flu = np.mean(flu_trial[:, :self.pre_frames], axis=1)
                baseline_flu_stack = np.repeat(baseline_flu, flu_trial.shape[1]).reshape(flu_trial.shape)
                flu_trial = flu_trial - baseline_flu_stack

            # only append trials of the correct length - will catch corrupt/incomplete data and not include
            if i == 0:
                trial_array = flu_trial
                flu_trial_shape = flu_trial.shape[1]
            else:
                if flu_trial.shape[1] == flu_trial_shape:
                    trial_array = np.dstack((trial_array, flu_trial))
                else:
                    print('**incomplete trial detected and not appended to trial_array**')
                    
        return trial_array
    
    
    def _trialProcessing(self, plane, pre_trial_frames, post_trial_frames):
        
        # make trial arrays from dff data [plane x cell x frame x trial]
        trial_array = self._makeFluTrials(self.dfof[plane], plane)
        self.all_trials.append(trial_array)
        self.n_trials = trial_array.shape[2]
                
        pre_array = np.mean(trial_array[:, pre_trial_frames, :], axis=1)
        post_array = np.mean(trial_array[:, post_trial_frames, :], axis=1)
        all_amplitudes = post_array - pre_array
        self.all_amplitudes.append(all_amplitudes)
        
        # significance test, [cell (p-value)]
#         t_tests = stats.ttest_rel(pre_array, post_array, axis=1)
#         self.t_tests.append(t_tests[1][:])
        wilcoxons = np.empty(self.n_units[0])

        for cell in range(self.n_units[0]):
            wilcoxons[cell] = stats.wilcoxon(post_array[cell], pre_array[cell])[1]

        self.wilcoxons.append(wilcoxons)
        
        return trial_array 
    
    
    def _STAProcessing(self, trial_array, pre_trial_frames, post_trial_frames):
        
        # make stas, [plane x cell x frame]
        stas = np.mean(trial_array, axis=2)
        self.stas.append(stas)

        # make sta amplitudes, [plane x cell]
        pre_sta = np.mean(stas[:, pre_trial_frames], axis=1)
        post_sta = np.mean(stas[:, post_trial_frames], axis=1)
        sta_amplitudes = post_sta - pre_sta
        self.sta_amplitudes.append(sta_amplitudes)
    
    
    def _sigTestTrialDFSF(self, plane, post_trial_frames):
        
        # make dF/sF trials to calculate single trial responses from
        df_trials = self._makeFluTrials(self.raw[plane], plane)   
        std_baseline = np.std(df_trials[:, :self.pre_frames, :], axis=1) # std of baseline period
        std_baseline = np.expand_dims(std_baseline, axis=1)
        std_baseline = np.repeat(std_baseline, df_trials.shape[1], axis=1)
        dfsf_trials = df_trials/std_baseline

        mean_post_dfsf = np.nanmean(dfsf_trials[:, post_trial_frames, :], axis=1)

        trial_sig = np.absolute(mean_post_dfsf > 1)
        
        self.trial_sig_dfsf.append(trial_sig)
        
    
    def _sigTestTrialDFF(self, plane):
        
        dff_baseline = self.all_trials[plane][:, :self.pre_frames, :] # [cell x frames x trial]
        std_baseline = np.std(dff_baseline, axis=1)

        trial_sig = np.absolute(self.all_amplitudes[plane]) >= 2*std_baseline # [cell x trial]
            
        # [plane x cell x trial]
        self.trial_sig_dff.append(trial_sig)
        
        
    def _sigTestAvgDFF(self, plane):

#         p_vals = self.t_tests[plane]

#         bonf_corr = [i for i,p in enumerate(p_vals) if p < 0.05 / self.n_units[plane]]
#         sig_units = np.zeros(self.n_units[plane], dtype='bool')
#         sig_units[bonf_corr] = True

        p_vals = self.wilcoxons[plane]
        
        sig_units, _, _, _ = smstats.multitest.multipletests(p_vals, alpha=0.1, method='fdr_bh', 
                                                             is_sorted=False, returnsorted=False)
        
        no_bonf_corr = [i for i,p in enumerate(p_vals) if p < 0.05]
        nomulti_sig_units = np.zeros(self.n_units[plane], dtype='bool')
        nomulti_sig_units[no_bonf_corr] = True

        self.sta_sig.append(sig_units)  
        self.sta_sig_nomulti.append(nomulti_sig_units)

        
    def _probResponse(self, plane, trial_sig_calc):
        
        # calculate response probability across all trials for each cell
        n_trials = self.n_trials

        # get the number of responses across all trials
        if trial_sig_calc == 'dff':
            num_respond = np.array(self.trial_sig_dff[plane]) # trial_sig_dff is [plane][cell][trial]
        elif trial_sig_calc == 'dfsf': 
            num_respond = np.array(self.trial_sig_dfsf[plane])
        
        # return the probability of response
        self.prob_response.append(np.sum(num_respond, axis=1) / n_trials)
    
    
    def _makeTimeVector(self):
        
        n_frames = self.all_trials[0].shape[1]
        pre_time = self.pre_frames/self.fps
        post_time = self.post_frames/self.fps
        self.time = np.linspace(-pre_time, post_time, n_frames)
        
    
    def _scaleTargets(self, frame_x, frame_y, target_image_scaled):
                          
        #bounding box coords
        x1 = 511 - frame_x/2
        x2 = 511 + frame_x/2
        y1 = 511 - frame_y/2
        y2 = 511 + frame_y/2

        #calc imaging galvo offset between BAFOV and t-series
        zoom = self.zoom
        scan_x = self.scan_x #scan centre in V
        scan_y = self.scan_y

        ScanAmp_X = 2.62*2
        ScanAmp_Y = 2.84*2

        ScanAmp_V_FOV_X = ScanAmp_X / zoom
        ScanAmp_V_FOV_Y = ScanAmp_Y / zoom

        scan_pix_y = ScanAmp_V_FOV_Y / 1024
        scan_pix_x = ScanAmp_V_FOV_X / 1024

        offset_x = scan_x/scan_pix_x #offset between image centres in pixels
        offset_y = scan_y/scan_pix_y

        # offset the bounding box
        x1,x2,y1,y2 = round(x1+offset_x), round(x2+offset_x), round(y1-offset_y), round(y2-offset_y)

        # crop the target image using the offset bounding box to get the targets in t-series imaging space
        return target_image_scaled[y1:y2, x1:x2]
        
                          
    def _findTargetAreas(self):
                          
        print('\nFinding SLM target locations...')
        
        # load naparm targets file for this experiment
        naparm_path = os.path.join(self.naparm_path, 'Targets')

        listdir = os.listdir(naparm_path)

        for path in listdir:
            if 'AllFOVTargets' in path:
                target_file = path

        target_image = tf.imread(os.path.join(naparm_path, target_file))

        n = np.array([[0, 0],[0, 1]], dtype='uint8')
        target_image_scaled = np.kron(target_image, n)

        # use frame_x and frame_y to get bounding box of OBFOV inside the BAFOV, assume BAFOV always 1024x1024
        frame_x = self.frame_x
        frame_y = self.frame_y

        if frame_x < 1024 or frame_y < 1024:
            target_image_scaled = self._scaleTargets(frame_x, frame_y, target_image_scaled)
            tf.imwrite(os.path.join(naparm_path, 'target_image_scaled.tif'), target_image_scaled)
        else:
            tf.imwrite(os.path.join(naparm_path, 'target_image_scaled.tif'), target_image_scaled)
        
        targets = np.where(target_image_scaled>0)

        targ_coords = list(zip(targets[0], targets[1]))
        print('number of targets:', len(targ_coords))

        self.target_coords = targ_coords
        self.n_targets = len(targ_coords)

        target_areas = []

        radius = int(((self.spiral_size/2)+10)/self.pix_sz_x) # adding 10 um for photostim res
        for coord in targ_coords:
            target_area = np.array([item for item in points_in_circle_np(radius, x0=coord[0], y0=coord[1])])
            if not any([max(target_area[:,1]) > frame_x,
                        max(target_area[:,0]) > frame_y,
                        min(target_area[:,1]) < 0,
                        min(target_area[:,0]) < 0
                       ]):
                target_areas.append(target_area)

        self.target_areas = target_areas
    
                          
    def _findTargetedCells(self):
                    
        print('searching for targeted cells...')

        targ_img = np.zeros([self.frame_x, self.frame_y], dtype='uint16')

        target_areas = np.array(self.target_areas)

        targ_img[target_areas[:,:,1], target_areas[:,:,0]] = 1
 
        cell_img = np.zeros_like(targ_img)

        cell_x = np.array(self.cell_x)
        cell_y = np.array(self.cell_y)

        for i,coord in enumerate(zip(cell_x[0], cell_y[0])):
            cell_img[coord] = i+1

        targ_cell = cell_img*targ_img

        targ_cell_ids = np.unique(targ_cell)[1:]-1

        self.targeted_cells = np.zeros([self.n_units[0]], dtype='bool')
        self.targeted_cells[targ_cell_ids] = True

        self.n_targeted_cells = np.sum(self.targeted_cells)

        print('search completed.')
        print('number of targeted cells: ', self.n_targeted_cells)
            
            
    def _targetSumDff(self):
        
        trial_amplitudes = self.all_amplitudes[0]

        targeted_cells = np.repeat(self.targeted_cells[..., None], 
                               trial_amplitudes.shape[1], 1) # [..., None] is a quick way to expand_dims

        trial_target_amplitudes = np.multiply(trial_amplitudes, targeted_cells)
        trial_target_dff = np.sum(trial_target_amplitudes, axis=0)

        return trial_target_dff
    
    
    def _euclidDist(self, resp_positions):
        
        # mean distance of targets from centroid
        resp_coords = list(zip(*resp_positions))
        centroidx = np.mean(resp_coords[0])
        centroidy = np.mean(resp_coords[1])
        centroid = [centroidx, centroidy]

        dists = np.empty(resp_positions.shape[0])

        for i, resp in enumerate(resp_positions):
            dists[i] = np.linalg.norm(resp - centroid)

        euclid_dist = np.mean(dists)

        return euclid_dist

    
    def _targetSpread(self):
        
        # Avg Euclidean dist of responding targeted cells on each trial
        trial_responders = self.trial_sig_dff[0]
        targeted_cells = np.repeat(self.targeted_cells[..., None], 
                                   trial_responders.shape[1], 1) # [..., None] is a quick way to expand_dims
        targeted_responders = targeted_cells & trial_responders

        cell_positions = np.array(self.cell_med[0])
        
        dists = np.empty(self.n_trials)
        
        for i, trial in enumerate(range(self.n_trials)):

            resp_cell = np.where(targeted_responders[:,trial])
            resp_positions = cell_positions[resp_cell]

            if resp_positions.shape[0] > 1: # need more than 1 cell to measure spread...
                dists[i] = self._euclidDist(resp_positions)
            else:
                dists[i] = np.nan

        self.trial_euclid_dist = dists

        # Avg Euclidean dist of consistently responding targeted cells (all trials)
        responder = self.sta_sig[0]
        target = self.targeted_cells

        targeted_responders = responder & target

        resp_cell = np.where(targeted_responders)
        resp_positions = cell_positions[resp_cell]

        if resp_positions.shape[0] > 1:
            dist = self._euclidDist(resp_positions)
        else: 
            dist = np.nan

        self.sta_euclid_dist = dist
    
    
    def _targetAnalysis(self):
        
        self._findTargetAreas()
        
        self._findTargetedCells()

        self.trial_target_dff = self._targetSumDff()
        
        self._targetSpread()

                          
    def s2pAnalysis(self, s2_borders_path, trial_sig_calc):
        
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
                    
                    pre_trial_frames = np.s_[self.pre_frames - self.test_frames : self.pre_frames]
                    stim_end = self.pre_frames + self.duration_frames
                    post_trial_frames = np.s_[stim_end : stim_end + self.test_frames]
        
                    trial_array = self._trialProcessing(plane, pre_trial_frames, post_trial_frames)

                    self._STAProcessing(trial_array, pre_trial_frames, post_trial_frames)
                
                    self._sigTestTrialDFSF(plane, post_trial_frames)
                    self._sigTestTrialDFF(plane)
                    self._sigTestAvgDFF(plane)
                    
                    self._probResponse(plane, trial_sig_calc)
                
                self._makeTimeVector()
                
            # target cells
            if any(s in self.stim_type for s in ['pr', 'ps', 'none']):
                self._targetAnalysis()
                          
                