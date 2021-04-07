import sys
import os
import numpy as np
import pickle
import json
from opto_stim_import2 import BlimpImport, OptoStim2p
sys.path.append(os.path.expanduser('~/Documents/code/suite2p'))  # make this not shit
import suite2p
from suite2p.run_s2p import run_s2p
from my_suite2p.settings import ops
import utils_funcs as utils
import run_functions as rf
import re
import tifffile
import glob
import ntpath
from pathlib import Path
import time
import traceback
import datetime

''' This should maybe become a class '''

def run_processor(mouse_id, run_number, pkl_path, 
                  reprocess=False, reload=False):

    mouse_folder = os.path.join(pkl_path, mouse_id)
    
    if not os.path.exists(mouse_folder):
        os.makedirs(mouse_folder)
    
    global pkl_file  # sorry
    pkl_file = os.path.join(mouse_folder, 
                            'run{}.pkl'.format(run_number))

    if os.path.isfile(pkl_file) and not reprocess:
        print('run number {} already processed\n'
              .format(run_number))
        
        if reload:
            with open(pkl_file, 'rb') as f:
                print('Reloading processed pkl') 
                run =  pickle.load(f)
                run.reloaded = True
            return run
        else:
            return None
        
    run = BlimpImport(mouse_id)
    run.reloaded = False

    try:
        run.get_object_and_test(run_number, raise_error = True)
        return run
        
    except Exception as e:
        # get the information about the exception
        print('\nError for run number {}\n'.format(run_number))
        var = traceback.format_exc()
        print(var)

        return None
        
def tiff_metadata(tiff_folder):

    ''' takes input of list of tiff folders and returns 
        number of frames in the first of each tiff folder '''

    with tifffile.TiffFile(tiff_folder) as tif:
        tif_tags = {}
        for tag in tif.pages[0].tags.values():
            name, value = tag.name, tag.value
            tif_tags[name] = value

    x_px = tif_tags['ImageWidth']
    y_px = tif_tags['ImageLength']
    image_dims = [x_px, y_px]

    n_frames = re.search('(?<=\[)(.*?)(?=\,)', 
                         tif_tags['ImageDescription'])

    n_frames = int(n_frames.group(0))

    return image_dims, n_frames


def preprocess_flu(run):

    ''' Function to load the fluoresence matrix from suite2p output,
        and use the rsync aligner to find the time in ms (relative
        to the pycontrol timer) that each frame occured on.

        Input:
        run -- BlimpImport object to be processed

        Returns BlimpImport Object with attributes:
        flu -- dfof fluoresence matrix of rois marked 
               as cells in s2p gui
        spks -- oasis deconv spks of cells marked in gui
        stat -- s2p info about cells marked in gui
        frames_ms -- matrix with same shape as flu, with 
                     the time in ms of each frame.
                     If nan, frame time did not fall between
                     two rsync pulses (likely because pycontrol
                     not running)
        '''

    if run.num_planes > 1:
        utils.correct_s2p_combined(run.s2p_path, run.num_planes)
        s2p_path = os.path.join(run.s2p_path, 'combined')
    else:
        s2p_path = os.path.join(run.s2p_path, 'plane0')

    print(s2p_path)
    flu_raw, _, _ = utils.s2p_loader(s2p_path, 
                                     subtract_neuropil=False) 
    run.flu_raw = flu_raw

    flu, spks, stat = utils.s2p_loader(s2p_path)
    flu = utils.dfof2(flu)

    num_planes = run.num_planes # the lengths of the actual tseries
    tseries_lens = run.num_frames
    # get the frame ttls recorded in paqio that actually 
    # correspond to analysed tseries
    paqio_frames = utils.tseries_finder(tseries_lens, 
                                        run.frame_clock)
    run.paqio_frames = paqio_frames

    if len(paqio_frames) == sum(tseries_lens):
        print('All tseries chunks found in frame clock')
    else:
        print('WARNING: Could not find all tseries chunks in '
              'frame clock, check this')
        print('Total number of frames detected in clock is {}'
               .format(len(paqio_frames)))
        print('These are the lengths of the tseries from '
               'spreadsheet {}'.format(tseries_lens))
        print('The total length of the tseries spreasheets is {}'
               .format(sum(tseries_lens)))
        missing_frames = sum(tseries_lens) - len(paqio_frames)
        print('The missing chunk is {} long'.format(missing_frames))
        try:
            print('A single tseries in the spreadsheet list is '
                  'missing, number {}'.format(tseries_lens.index
                                             (missing_frames) + 1))
        except ValueError:
            print('Missing chunk cannot be attributed to a single '
                   'tseries')

    if flu.shape[1] * num_planes >= len(paqio_frames) and \
       flu.shape[1] * num_planes <= len(paqio_frames) + num_planes:
        print('Fluorescence matrix shape matches recorded frame '
              'clock')
    else:
        print('WARNING: Fluoresence matrix does not match frame '
               'clock shape, check this')

    # which plane is each cell in
    try:
        cell_plane = np.array([s['iplane'] for s in stat])
    except KeyError:
        cell_plane = np.zeros(len(stat))

    ##### Don't think we need this anymoree

    # run.frames_ms = build_frames_ms(run, cell_plane, paqio_frames, 
                                    # aligner=run.aligner)

    # run.frames_ms_pre = build_frames_ms(run, cell_plane, 
                                        # paqio_frames, aligner =
                                        # run.prereward_aligner)

    # add to the run object
    run.flu = flu
    run.spks = spks
    run.stat = stat

    # Add in stuff from the monkey patcher notebook
    gt = rf.GetTargets(run)
    run.is_target = gt.is_target

    run = rf.spiral_tstart(run)
    run.spiral_licks = rf.get_binned_licks(run, run.aligner.B_to_A(run.spiral_start))
    run.autorewarded_trial = rf.autoreward(run) 

    return run

def build_frames_ms(run, cell_plane, paqio_frames, aligner):

    ''' builds frames_ms matrix (see preprocess_flu)
        aligner -- rsync object from rsync_aligner
   
        '''
    # convert paqio recorded frames to pycontrol ms
    ms_vector = aligner.B_to_A(paqio_frames) # flat list of plane 
                                             # times
    if run.num_planes == 1:
        return ms_vector
    
    # matrix of frame times in ms for each fluorescent value 
    # in the flu matrix
    frames_ms = np.empty(run.flu_raw.shape)
    frames_ms.fill(np.nan)
 
    # mark each frame with a time in ms based on its plane
    for plane in range(run.num_planes):
        frame_times = ms_vector[plane::run.num_planes]
        plane_idx = np.where(cell_plane==plane)[0]
        frames_ms[plane_idx, 0:len(frame_times)] = frame_times

    return frames_ms

def main(mouse_id, run_number, pkl_path,
         do_s2p=False, reprocess=True, 
         reload=True, do_flu_preprocess=True):

    ''' caches an single session for one mouse to pkl'''
        
    run = run_processor(mouse_id, run_number, pkl_path, 
                        reprocess=reprocess, reload=reload)
    
    if run is None:
        return

    # flatten out the tseries path lists if it is nested
    if any(isinstance(i, list) for i in run.tseries_paths):
        run.tseries_paths = [item for sublist in run.tseries_paths
                             for item in sublist]
    tseries_nframes = []
    tseries_dims = []
    
    print('\nfollowing tseries found:')
    tiff_list = []
    print(run.tseries_paths)
    for tseries in run.tseries_paths:
        tiffs = utils.get_tiffs(tseries)
        if not tiffs:
            raise filenotfounderror('cannot find tiff in '
                                     'folder {}'.format(tseries))
        elif len(tiffs) == 1:
            assert tiffs[0][-7:] == 'Ch3.tif', 'channel not understood '\
                                              'for tiff {}'.format(tiff)
                                       
            tiff_list.append(tiffs[0])
        elif len(tiffs) == 2:  # two channels recorded (red is too dim)
            assert tiffs[0][-7:] == 'Ch2.tif' and tiffs[1][-7:] == 'Ch3.tif',\
                                        'channel not understood '\
                                        'for tiffs {} and {}'.format(tiffs[0],
                                                                        tiffs[1])

            tiffs = [tiffs[1]]
            tiff_list.append(tiffs[0])

        elif len(tiffs) > 2:
            print('Single frame tiffs not deleted, checking for MPTs')

             # Check if multipage tiff conversion happend but single tiffs
            # were not deleted
            multipages = [tiff for tiff in tiffs if not '.ome' in tiff]
            multipages.sort()

            if not multipages:
                raise ValueError('Folder likely contains unconverted single '
                                 'page tiffs')

            # check that the number of tiffs in the multipage matches the 
            # number of .ome
            num_ome = len([tiff for tiff in tiffs  if '.ome' in tiff])
            for i, multipage in enumerate(multipages):
                _, n_frames = tiff_metadata(multipage)
                # ideally want equal but sometimes deletion starts but is 
                # not completed
                assert n_frames >= num_ome or n_frames == num_ome/len(multipages), \
                '{} {} {}'.format(multipage, n_frames, num_ome)
            # check for the green channel
            if len(multipages) > 1:
                assert multipages[1][-7:] == 'Ch3.tif', 'channel not understood '\
                                             'for tiff {}'.format(multipages)
                tiff_list.append(multipages[1])
                tiffs = [multipages[1]]

            elif len(multipages) == 1:
                assert multipages[0][-7:] == 'Ch3.tif', 'channel not understood '\
                                             'for tiff {}'.format(multipages)
                
                tiff_list.append(multipages[0])
                tiffs = [multipages[0]]

        image_dims, n_frames = tiff_metadata(tiffs[0])
        tseries_dims.append(image_dims)
        tseries_nframes.append(n_frames)

    print('\n') 

    run.num_frames = tseries_nframes

    print(tseries_dims)
    if tseries_dims[0][0] == 1024 and tseries_dims[0][1] == 1024: 
        diameter = 11
        fs = 15
        print('Bafov detected')
        raise  # path hack in db has ruined bafov
    elif tseries_dims[0][0] == 1024 and tseries_dims[0][1] == 514:
        diameter = 11
        fs = 30
        assert run.num_planes == 1  # temporary
        print('obfov detected')
    else:
        raise NotImplementedError('Cacher is currently only set up for bafov '
                                  'and obfov images')

    data_path = str(Path(run.tseries_paths[0]).parent)
    save_folder = os.path.join(data_path, 'suite2p', run.mouse_id)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    db = {
       'data_path' : [data_path],
       'look_one_level_down' : True,
       'diameter'  : diameter, 
       'tiff_list' : tiff_list,
       'nplanes'   : run.num_planes,
       'fs'        : fs,
       'save_folder': save_folder,
       'roidetect': True,
       'spikedetect': False,
        }   

    # TEMPORARRY??
    run.s2p_path = os.path.join(Path(db['data_path'][0]),
                                'suite2p')
    # 2021 addition, think this is correct
    run.s2p_path = save_folder


#    # check that suite2p hasn't alreay been run
#    if os.path.exists(run.s2p_path):
#        print('Already done s2p\n')
#        do_s2p = False

    print('Data path is {}'.format(data_path))
    print('s2p path is {}'.format(run.s2p_path))

    if do_s2p:
        # Build bad_frames.npy (not sure which directory to use so do both
        # useful to keep record of bad frames in mouse specific suite2p 
        # output folder
        rf.get_bad_frames(run, [data_path, run.s2p_path, save_folder])
        print('Running s2p on tseries printed above\n')
        with open('/home/jrowland/mnt/qnap/suite2p_report.txt', 'a') as f:
            dateTime = datetime.datetime.today()
            f.write('\n{} Beginning s2p for {} run {} path {}'
                    .format(dateTime.isoformat("|", 'seconds'), run.mouse_id, 
                            run_number, save_folder))
            
        opsEnd=run_s2p(ops=ops,db=db)

        with open('/home/jrowland/mnt/qnap/suite2p_report.txt', 'a') as f:
            dateTime = datetime.datetime.today()
            f.write('\n{} Completed s2p for {} run {} path {}\n\n'
                    .format(dateTime.isoformat("|", 'seconds'), run.mouse_id, 
                            run_number, save_folder))

    if do_flu_preprocess:
        run  = preprocess_flu(run)

    if not run.reloaded:
        with open(pkl_file, 'wb') as f:
            pickle.dump(run, f)

