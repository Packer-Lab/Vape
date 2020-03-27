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
from scipy import stats

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
        print('\nFetching paths and stim types for:' , sheet_name)
        
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
        print('Umbrella folder:', umbrella)

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
        print('Experimental info =', info)

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
        self.spont.single_stim_dur = self.photostim_r.single_stim_dur
        self.spont.n_shots = self.photostim_r.n_shots
        self.spont.n_groups = self.photostim_r.n_groups
        self.spont.n_trials = self.photostim_r.n_trials
        self.spont.inter_point_delay = self.photostim_r.inter_point_delay
        self.spont.targeted_cells = self.photostim_r.targeted_cells
        self.spont.n_targets = self.photostim_r.n_targets

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

            print('\nObtaining metadata for', self.stim_type, 'stim:', self.tiff_path)
            self._parsePVMetadata()
                        
            self.stimProcessing()
        
        else:
            self.stim_type = stim_type
            
            self.tiff_path = None
            self.naparm_path = None
            self.paq_path = None

            self.n_frames = 0

            print('\nNo metadata for this', self.stim_type, 'stim experiment')
    
    
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

        self.fps = fps
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
            '\nfps:', fps,
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

        print('Numbers of trials:', n_trials,
            '\nInter-group delay:', inter_point_delay,
            '\nNumber of groups:', n_groups,
            '\nNumber of shots:', n_shots,
            '\nNumber of sequence reps:', n_reps,
            )

        self.n_groups = n_groups
        self.n_reps = n_reps
        self.n_shots = n_shots
        self.n_trials = n_trials
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

        # find frame times
        clock_idx = paq['chan_names'].index('frame_clock')
        clock_voltage = paq['data'][clock_idx, :]

        frame_clock = threshold_detect(clock_voltage, 1)

        # find stim times
        stim_idx = paq['chan_names'].index(self.stim_channel)
        stim_volts = paq['data'][stim_idx, :]
        stim_times = threshold_detect(stim_volts, 1)

        #correct this based on txt file
        duration_ms = self.stim_dur
        frame_rate = self.fps/self.n_planes
        duration_frames = np.ceil((duration_ms/1000)*frame_rate)
        self.duration_frames = int(duration_frames)
        print('total stim duration (frames):', int(duration_frames))

        # find stim frames
        self.stim_start_frames = []

        for plane in range(self.n_planes):
            
            stim_start_frames = []

            for stim in stim_times:

                #the index of the frame immediately preceeding stim
                stim_start_frame = next(i-1 for i,sample in enumerate(frame_clock[plane::self.n_planes]) if sample - stim >= 0)
                stim_start_frames.append(stim_start_frame)
                
            self.stim_start_frames.append(stim_start_frames)
        
        #sanity check - reduce stim_start_frames until it is below n_frames to catch missing frames
        while np.amax(self.stim_start_frames) > self.n_frames:
            for plane in range(self.n_planes):
                self.stim_start_frames[plane] = self.stim_start_frames[plane][:-1]


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

        self.stim_dur = 1000 # total whisker stim duration
        self.paqProcessing()
        self.duration_frames = 0 # don't need to exclude stim artifact for whisker_stim
        
        # calculate number of trials that could fit in the t-series
        # only really applicable for t-series with data loss at end of t-series
        trial_length = np.diff(self.stim_start_frames)
        trial_length_count = np.bincount(trial_length[0])
        mode_trial_length = np.argmax(trial_length_count)
        max_n_trials = (self.n_frames - self.stim_start_frames[0][0])/mode_trial_length
        
        if max_n_trials < 100:
            self.n_trials = int(floor(max_n_trials))
        else:
            self.n_trials = len(self.stim_start_frames[0])
            
        
    def stimProcessing(self):

        if self.stim_type == 'w':
            self.whiskerStimProcessing()
        if any(s in self.stim_type for s in ['pr', 'ps']):
            self.photostimProcessing()

            
    def _saveS2pMasks(self, cell_ids, save_name):

        #create and save the s2p cell masks, the targeted cell masks and the mean image
        s2p_path = self.s2p_path
        exp_name = os.path.basename(self.tiff_path)

        os.chdir(s2p_path)

        stat = np.load('stat.npy', allow_pickle=True)
        ops = np.load('ops.npy', allow_pickle=True).item()
        iscell = np.load('iscell.npy', allow_pickle=True)           

        im = np.zeros((ops['Ly'], ops['Lx']), dtype='uint16')

        for n in range(0,len(iscell)):
            if n in cell_ids:
                ypix = stat[n]['ypix']
                xpix = stat[n]['xpix']
                im[ypix,xpix] = n

        tf.imwrite(exp_name + '_' + save_name + '.tif', im)
                        
        print('Done')
    
    
    def _saveMeanImage(self):
        
        print('Saving mean image...')
        
        s2p_path = self.s2p_path
        exp_name = os.path.basename(self.tiff_path)

        os.chdir(s2p_path)
        
        ops = np.load('ops.npy', allow_pickle=True).item()
        
        mean_img = ops['meanImg']

        mean_img = np.array(mean_img, dtype='uint16')
        tf.imwrite(exp_name + '_mean_image.tif', mean_img)
        
        
    def _findTargets(self):

        self.n_targets = []
        self.target_coords = []
        self.target_areas = []
        self.targeted_cells = []
        
        # load naparm targets file for this experiment
        naparm_path = os.path.join(self.naparm_path, 'Targets')

        listdir = os.listdir(naparm_path)

        for path in listdir:
            if 'AllFOVTargets' in path:
                target_file = path

        target_image = tf.imread(os.path.join(naparm_path, target_file))

        n = np.array([[0, 0],[0, 1]])
        target_image_scaled = np.kron(target_image, n)

        # use frame_x and frame_y to get bounding box of OBFOV inside the BAFOV, assume BAFOV always 1024x1024
        frame_x = self.frame_x
        frame_y = self.frame_y

        if frame_x < 1024 or frame_y < 1024:
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
            target_image_scaled = target_image_scaled[y1:y2, x1:x2]
            tf.imwrite(os.path.join(naparm_path, 'target_image_scaled.tif'), target_image_scaled)
        else:
            tf.imwrite(os.path.join(naparm_path, 'target_image_scaled.tif'), target_image_scaled)
        
        targets = np.where(target_image_scaled>0)

        targetCoordinates = list(zip(targets[1], targets[0]))
        print('Number of targets:', len(targetCoordinates))
                
        self.target_coords = targetCoordinates
        self.n_targets = len(targetCoordinates)

        target_areas = []

        radius = self.spiral_size/self.pix_sz_x # this is effectively double the spiral size
        for coord in targetCoordinates:
            target_area = ([item for item in points_in_circle_np(radius, x0=coord[0], y0=coord[1])])
            target_areas.append(target_area)

        self.target_areas = target_areas

        print('Searching for targeted cells...')

        for cell in range(self.n_units[0]):
            flag = 0
            
            for x, y in zip(self.cell_x[0][cell], self.cell_y[0][cell]):
                for target in range(self.n_targets):
                    for a, b in self.target_areas[target]:
                        if x == a and y == b:
                            flag = 1

            if flag==1:
                self.targeted_cells.append(1)
            else:
                self.targeted_cells.append(0)

        self.n_targeted_cells = len([i for i in self.targeted_cells if i == 1])

        print('Search completed.')
        print('Number of targeted cells: ', len([i for i in self.targeted_cells if i == 1]))

        
    def _staSignificance(self, test):
        
        self.sta_sig = []
        self.sta_sig_nomulti = []
        
        for plane in range(self.n_planes):
            
            #set this to true if you want to multiple comparisons correct for the number of cells
            multi_comp_correction = True
            if not multi_comp_correction: 
                divisor = 1
            else:
                divisor = self.n_units[plane]

            if test is 't_test':
                p_vals = [t[1] for t in self.t_tests[plane]]
            if test is 'wilcoxon':
                p_vals = [t[1] for t in self.wilcoxons[plane]]

            if multi_comp_correction:
                print('performing t-test on cells with multiple comparisons correction')
            else:
                print('performing t-test on cells without mutliple comparisons correction')

            sig_units = []
            nomulti_sig_units = []
            
            for _,p in enumerate(p_vals):
                # unit_index = self.cell_id[plane][i]

                if p < 0.05:
                    nomulti_sig_units.append(True)
                else:
                    nomulti_sig_units.append(False)
                if p < (0.05 / divisor):
                    sig_units.append(True) #significant units
                else:
                    sig_units.append(False) 

            self.sta_sig.append(sig_units)  
            self.sta_sig_nomulti.append(nomulti_sig_units)
    
    
    def _singleTrialSignificance(self):
        
        self.single_sig = [] # single trial significance value for each trial for each cell in each plane

        for plane in range(self.n_planes):

            single_sigs = []
            
            for cell,_ in enumerate(self.cell_id[plane]):
                
                single_sig = []

                for trial in range(len(self.all_trials[0][0])):
                    
                    pre_f_trial  = self.all_trials[plane][cell][trial][ : self.pre_frames ]
                    std = np.std(pre_f_trial)

                    if np.absolute(self.all_amplitudes[plane][cell][trial]) >= 2*std:
                        single_sig.append(True)
                    else:
                        single_sig.append(False)
                
                single_sigs.append(single_sig)

            self.single_sig.append(single_sigs)

            
    def _cellStaProcessing(self, test='t_test'):
        
        if self.stim_start_frames:
            
            #this is the key parameter for the sta, how many frames before and after the stim onset do you want to use
            self.pre_frames = int(np.ceil(self.fps*0.5)) # 500 ms pre-stim period
            self.post_frames = int(np.ceil(self.fps*3)) # 3000 ms post-stim period

            #list of cell pixel intensity values during each stim on each trial
            self.all_trials = [] # list 1 = cells, list 2 = trials, list 3 = dff vector

            # the average of every trial
            self.stas = [] # list 1 = cells, list 2 = sta vector

            self.all_amplitudes = []
            self.sta_amplitudes = []

            self.t_tests = []
            self.wilcoxons = []

            for plane in range(self.n_planes):
                
                raw = self.raw[plane][:,self.frames] 

                all_trials = [] # list 1 = cells, list 2 = trials, list 3 = dff vector

                stas = [] # list 1 = cells, list 2 = sta vector

                all_amplitudes = []
                sta_amplitudes = []

                t_tests = []
                wilcoxons = []

                #loop through each cell
                for i, unit in enumerate(raw):

                    trials = []
                    amplitudes = []
                    df = []
                    
                    # a flat list of all observations before stim occured
                    pre_obs = []
                    # a flat list of all observations after stim occured
                    post_obs = []
                    
                    for stim in self.stim_start_frames[plane]:
                        
                        # get baseline values from pre_stim
                        pre_stim_f  = unit[ stim - self.pre_frames : stim ]
                        baseline = np.mean(pre_stim_f)

                        # the whole trial and dfof using baseline
                        trial = unit[ stim - self.pre_frames : stim + self.post_frames ]
                        trial = [ ( (f-baseline) / baseline) * 100 for f in trial ] #dff calc
                        
                        if trial: #this is because of missing data in t-series, can be removed RL
                            
                            trials.append(trial)
                            
                            #calc amplitude of response        
                            pre_f = trial[ : self.pre_frames - 1]
                            pre_f = np.mean(pre_f)
                            
                            avg_post_start = self.pre_frames + ( self.duration_frames + 1 )
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
                    sta_amplitude = np.mean(amplitudes, axis=0)
                    sta_amplitudes.append(sta_amplitude)

                    #average across all trials
                    sta = np.mean(trials, axis=0)        
                    stas.append(sta)
                    
                    #remove nans from flat lists
                    pre_obs = [x for x in pre_obs if ~np.isnan(x)]
                    post_obs = [x for x in post_obs if ~np.isnan(x)]
                    
                    #t_test and man whit test pre and post stim (any other test could also be used here)
                    t_test = stats.ttest_rel(pre_obs, post_obs)
                    t_tests.append(t_test)
                    
#                     wilcoxon = stats.wilcoxon(pre_obs, post_obs)
#                     wilcoxons.append(wilcoxon)

                self.all_trials.append(np.array(all_trials))
                self.stas.append(np.array(stas))
                
                self.all_amplitudes.append(np.array(all_amplitudes))
                self.sta_amplitudes.append(np.array(sta_amplitudes))

                self.t_tests.append(np.array(t_tests))
                self.wilcoxons.append(np.array(wilcoxons))
            
#             plt.figure()
#             plt.plot([avg_post_start] * 2, [-1000, 1000])
#             plt.plot([avg_post_end] * 2, [-1000, 1000])
#             plt.plot([self.pre_frames - 1] * 2, [-1000, 1000])
#             plt.plot([0] * 2, [-1000, 1000])
#             plt.plot(stas[5])
#             plt.plot(stas[10])
#             plt.plot(stas[15])
#             plt.ylim([-100,200]) 

            self._staSignificance(test)
            self._singleTrialSignificance()
    
    
    def _findS1S2(self, cell_med, s2_borders_path):
    
        y = cell_med[0]
        x = cell_med[1]

        #delete this, otherwise it'll be called every time, add as argument somewhere
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
    
    
    def s2pAnalysis(self, s2_borders_path, subtract_neuropil=True):
        
        if self.n_frames == 0:
            print('\nNo data for this ', self.stim_type, ' experiment type')

        else: 
            print('\nProcessing s2p data for this ', self.stim_type, ' experiment type')

            self.cell_id = []
            self.n_units = []
            self.cell_plane = []
            self.cell_med = []
            self.cell_s1 = []
            self.cell_x = []
            self.cell_y = []
            self.raw = []
            self.mean_img = []

            for plane in range(self.n_planes):
                s2p_path = self.s2p_path
                FminusFneu, _, stat = s2p_loader(s2p_path, subtract_neuropil=True, neuropil_coeff=0.7)
                ops = np.load(os.path.join(s2p_path,'ops.npy'), allow_pickle=True).item()

                self.raw.append(FminusFneu)
                self.mean_img.append(ops['meanImg'])
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
                self.cell_x.append(cell_x)
                self.cell_y.append(cell_y)

                num_units = FminusFneu.shape[0]
                cell_plane.extend([plane]*num_units)
                self.cell_plane.append(cell_plane)

            self._cellStaProcessing()
            
            if any(s in self.stim_type for s in ['pr', 'ps', 'none']):
                self._findTargets()
            
                # Save s2p cell masks for different identities of cells
                #targets
                cell_ids = [self.cell_id[0][i] for i,b in enumerate(self.targeted_cells) if b==1]
                save_name = 'target_cell_masks'
                print('Creating and savings target cell masks from s2p...')
                self._saveS2pMasks(cell_ids, save_name)
            
            #whisker responsive cells
            if self.stim_type == 'w':
                cell_ids = [self.cell_id[0][i] for i,b in enumerate(self.sta_sig[0]) if b==1]
                save_name = 'whisker_responsive_masks'
                print('Creating and savings whisker responsive cell masks from s2p...')
                self._saveS2pMasks(cell_ids, save_name)
                
            #all cells
            cell_ids = self.cell_id[0]
            save_name = 'cell_masks'
            print('Creating and savings all cell masks from s2p...')
            self._saveS2pMasks(cell_ids, save_name)
            
            #s2 cells
            cell_ids = [self.cell_id[0][i] for i,b in enumerate(self.cell_s1[0]) if b==False]
            save_name = 's2_masks'
            print('Creating and savings s2 cell masks from s2p...')
            self._saveS2pMasks(cell_ids, save_name)
                        
            self._saveMeanImage()