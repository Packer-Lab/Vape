import numpy as np
import json
import tifffile as tf
import os
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import math
import bisect
import copy
import pickle

# global plotting params
params = {'legend.fontsize': 'x-large',
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
plt.rcParams.update(params)
sns.set()
sns.set_style('white')


def points_in_circle_np(radius, x0=0, y0=0, ):
    x_ = np.arange(x0 - radius - 1, x0 + radius + 1, dtype=int)
    y_ = np.arange(y0 - radius - 1, y0 + radius + 1, dtype=int)
    x, y = np.where((x_[:,np.newaxis] - x0)**2 + (y_ - y0)**2 <= radius**2)
    for x, y in zip(x_[x], y_[y]):
        yield x, y

        
def staMovie(output_dir, pkl_list=False):
    
    # OLD
    plane = 0
    exp_list = []

    for pkl_file in pkl_list:
            
        with open(pkl_file, 'rb') as f:
            ses_obj = pickle.load(f)
                
        exp_list = [ses_obj.photostim_r, ses_obj.photostim_s]

        if ses_obj.spont.n_frames > 0:
            exp_list.append(ses_obj.spont)

        if ses_obj.whisker_stim.n_frames > 0:
            exp_list.append(ses_obj.whisker_stim)

        for exp_obj in exp_list:
                
            print('\nMaking STA movie for:', exp_obj.tiff_path)

            size_y = exp_obj.frame_y
            size_x = exp_obj.frame_x
            size_z = (exp_obj.pre_frames*2) + int(exp_obj.stim_dur/exp_obj.fps)
                
            trial_stack = np.empty([0, size_z, size_y, size_x])
                
            for file in os.listdir(exp_obj.tiff_path):
                if '.tif' in file:
                    tiff_file = os.path.join(exp_obj.tiff_path, file)
                    break
                
            for t in range(exp_obj.n_trials):
                frame_start = exp_obj.stim_start_frames[plane][t]
                trial_start = frame_start - exp_obj.pre_frames
                trial_end = frame_start + exp_obj.pre_frames + int(exp_obj.stim_dur/exp_obj.fps)

                if trial_end <= exp_obj.n_frames: # for if the tiff file is incomplete (due to corrupt data)
                    trial = tf.imread(tiff_file, key=range(trial_start, trial_end))
                    trial = np.expand_dims(trial,axis=0)
                    trial_stack = np.append(trial_stack, trial, axis=0)
                        
            trial_avg = np.mean(trial_stack, axis=0)
            avg_baseline = trial_avg[: exp_obj.pre_frames, :, :]
            baseline_mean = np.mean(avg_baseline, 0)

            df_stack = trial_avg - baseline_mean                        
            dff_stack = (df_stack/baseline_mean) * 100
            dff_stack = dff_stack.astype('uint32')

            output_path = os.path.join(output_dir, file + '_plane' + str(plane) + '.tif')

            tf.imwrite(output_path, dff_stack)
            print('STA movie made for', np.shape(trial_stack)[0], 'trials:', output_path)
            
    
def cellFluTime(pkl_list):
        
    fig, ax = plt.subplots(nrows=len(pkl_list), ncols=1, figsize=(10,3*len(pkl_list)), sharex=True)

    for i,pkl in enumerate(pkl_list):
            
            print('Measuring mean cell fluorescence for:', pkl, '              ', end='\r')
            
            with open(pkl, 'rb') as f:
                ses_obj = pickle.load(f)

            mean_f = np.concatenate((np.mean(ses_obj.photostim_r.raw[0], axis=0),
                                     np.mean(ses_obj.photostim_s.raw[0], axis=0))
                                    )
            
            if ses_obj.whisker_stim.n_frames > 0:
                mean_f = np.concatenate((mean_f, np.mean(ses_obj.whisker_stim.raw[0], axis=0)))
                                    
            if ses_obj.spont.n_frames > 0:
                mean_f = np.concatenate((mean_f, np.mean(ses_obj.spont.raw[0], axis=0)))

            count = 0

            for frames in ses_obj.frame_list:
                x = range(count,count+frames)
                if len(pkl_list) > 1:
                    ax[i].plot(x, mean_f[x]);
                    ax[i].set_title(pkl.split('/')[-1])
                    ax[i].set_ylabel('mean_raw_f')
                else:
                    ax.plot(x, mean_f[x]);
                    ax.set_title(pkl.split('/')[-1])
                    ax.set_ylabel('mean_raw_f')
                    
                count += frames

    plt.xlabel('frames')
    labels = ['ps_random', 'ps_similar', 'whisker_stim', 'spont']
    plt.legend(labels)
    
    print('\nPlotting mean cell fluorescence...')
        
        
def frameFluTime(pkl_list, data_folder, legend=False):
        
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15,10), sharex=True)
    labels = []

    for _, _, files in os.walk(data_folder):
        for file in files:
            file_path = os.path.join(data_folder,file)

            tiff = tf.imread(file_path)

            raw_f_drift = np.mean(tiff, axis=(1,2)) - (2**16/2)
            norm_f_drift = raw_f_drift/raw_f_drift[0]

            ax[0].plot(raw_f_drift)
            ax[1].plot(norm_f_drift)
            labels.append(file)

    plt.xlabel('experiments');
    plt.xticks(range(0,8), np.tile(np.array(['start','end']), 4));
    ax[0].set_ylabel('raw_f');
    ax[1].set_ylabel('norm_f');
    ax[1].set_ylim([0.6, 1.2]);
    
    if legend:
        plt.legend(labels);
        
        
def downsampleTiff(pkl_list, save_path):
    
    for pkl in pkl_list:

        print('Downsampling experiment:', pkl, '             ', end='\r')
        
        with open(pkl, 'rb') as f:
            ses_obj = pickle.load(f)
        
        exp_list = [ses_obj.photostim_r, ses_obj.photostim_s]

        if ses_obj.spont.n_frames > 0:
            exp_list.append(ses_obj.spont)

        if ses_obj.whisker_stim.n_frames > 0:
            exp_list.append(ses_obj.whisker_stim)
        
        for exp_obj in exp_list:
            
            tiff_path = exp_obj.tiff_path
            
            file_list = os.listdir(tiff_path)
            for file in file_list:
                if '.tif' in file:
                    tiff_file = os.path.join(tiff_path, file)
  
            total_frames = range(0,exp_obj.n_frames) #get the range of frames for this experiment
            start_frames = total_frames[:1000] 
            end_frames = total_frames[-1000:] 
            
            stack_start = tf.imread(tiff_file, key=start_frames)
            stack_end = tf.imread(tiff_file, key=end_frames)
            
            mean_start = np.mean(stack_start, axis=0)
            mean_end = np.mean(stack_end, axis=0)
            
            output_path = os.path.join(save_path, tiff_path.split('/')[-1])
            
            tf.imsave(output_path + '_mean_start.tif', mean_start.astype('int16'))
            tf.imsave(output_path + '_mean_end.tif', mean_end.astype('int16'))
    
    
def s2pMeanImage(s2p_path):
        
    os.chdir(s2p_path)
        
    ops = np.load('ops.npy', allow_pickle=True).item()
        
    mean_img = ops['meanImg']

    mean_img = np.array(mean_img, dtype='uint16')
    
    return mean_img
    
    
def s2pMasks(s2p_path, cell_ids):
    
    os.chdir(s2p_path)

    stat = np.load('stat.npy', allow_pickle=True)
    ops = np.load('ops.npy', allow_pickle=True).item()
    iscell = np.load('iscell.npy', allow_pickle=True)           

    mask_img = np.zeros((ops['Ly'], ops['Lx']), dtype='uint16')

    for n in range(0,len(iscell)):
        if n in cell_ids:
            ypix = stat[n]['ypix']
            xpix = stat[n]['xpix']
            mask_img[ypix,xpix] = n

    return mask_img


def getTargetImage(obj):
    
    targ_img = np.zeros((obj.frame_y, obj.frame_x), dtype='uint16')
    targ_areas = obj.target_areas
        
    for targ_area in targ_areas:
        for coord in targ_area:
            targ_img[coord[0], coord[1]] = 255
        
    return targ_img

    
def s2pMaskStack(pkl_list, stam_save_path, parent_folder):
        
    for pkl in pkl_list:
                
        print('Retrieving s2p masks for:', pkl, '             ', end='\r')
            
        with open(pkl, 'rb') as f:
            ses_obj = pickle.load(f)
        
        # list of cell ideas to filter s2p masks by
        cell_id_list = [list(range(1,99999)), # all
                        ses_obj.photostim_r.cell_id[0], # cells
                        [ses_obj.photostim_r.cell_id[0][i] for i,b in enumerate(ses_obj.photostim_r.cell_s1[0]) if b==False], # s2 cells
                        [ses_obj.photostim_r.cell_id[0][i] for i,b in enumerate(ses_obj.photostim_r.targeted_cells) if b==1], # pr cells
                        [ses_obj.photostim_s.cell_id[0][i] for i,b in enumerate(ses_obj.photostim_s.targeted_cells) if b==1], # ps cells
                       ]
        
        # add whisker_stim if it exists for this session
        if ses_obj.whisker_stim.n_frames > 0:
            cell_id_list.append([ses_obj.whisker_stim.cell_id[0][i] for i,b in enumerate(ses_obj.whisker_stim.sta_sig[0]) if b==1]) # whisker cells
            
            for file in os.listdir(stam_save_path):
                if all(s in file for s in ['AvgImage', ses_obj.whisker_stim.tiff_path.split('/')[-1]]):
                    w_sta_img = tf.imread(os.path.join(stam_save_path, file))
                    w_sta_img = np.expand_dims(w_sta_img, axis=0)

        # empty stack to fill with images
        stack = np.empty((0, ses_obj.photostim_r.frame_y, ses_obj.photostim_r.frame_x), dtype='uint16')
        
        s2p_path = ses_obj.s2p_path
        
        # mean image from s2p
        mean_img = s2pMeanImage(s2p_path)
        mean_img = np.expand_dims(mean_img, axis=0)
        stack = np.append(stack, mean_img, axis=0)
        
        # mask images from s2p
        for cell_ids in cell_id_list:
            mask_img = s2pMasks(s2p_path, cell_ids)
            mask_img = np.expand_dims(mask_img, axis=0)
            stack = np.append(stack, mask_img, axis=0)
        
        # sta images
        for file in os.listdir(stam_save_path):
            if all(s in file for s in ['AvgImage', ses_obj.photostim_r.tiff_path.split('/')[-1]]):
                pr_sta_img = tf.imread(os.path.join(stam_save_path, file))
                pr_sta_img = np.expand_dims(pr_sta_img, axis=0)
            elif all(s in file for s in ['AvgImage', ses_obj.photostim_s.tiff_path.split('/')[-1]]):
                ps_sta_img = tf.imread(os.path.join(stam_save_path, file))
                ps_sta_img = np.expand_dims(ps_sta_img, axis=0)
        
        stack = np.append(stack, w_sta_img, axis=0)
        stack = np.append(stack, pr_sta_img, axis=0)
        stack = np.append(stack, ps_sta_img, axis=0)
        
        # target images
        pr_targ_img = getTargetImage(ses_obj.photostim_r)
        pr_targ_img = np.expand_dims(pr_targ_img, axis=0)
        stack = np.append(stack, pr_targ_img, axis=0)
        
        ps_targ_img = getTargetImage(ses_obj.photostim_s)
        ps_targ_img = np.expand_dims(ps_targ_img, axis=0)        
        stack = np.append(stack, ps_targ_img, axis=0)
        
        # stack is now: mean_img, all_rois, all_cells, s2_cells, pr_cells, ps_cells, 
        # (whisker,) pr_sta_img, ps_sta_img, pr_target_areas, ps_target_areas
        c,y,x = stack.shape
        stack.shape = 1, 1, c, y, x, 1 # dimensions in TZCYXS order
        
        x_pix = ses_obj.photostim_r.pix_sz_x
        y_pix = ses_obj.photostim_r.pix_sz_y
        
        save_path = os.path.join(parent_folder, pkl.split('/')[-1][:-4] + '_s2p_masks.tif')
        
        tf.imwrite(save_path, stack, imagej=True, resolution=(1/y_pix, 1/x_pix), photometric='minisblack')
            
def combineIscell(s2p_path, extra_iscell_path):
    
    os.chdir(s2p_path)

    iscell = np.load('iscell.npy', allow_pickle=True) 
    iscell_original = np.where(iscell[:,0])[0]
    
    iscell_extra = np.load(extra_iscell_path, allow_pickle=True)
    iscell_extra = np.where(iscell_extra[:,0])[0]

    iscell_combined = np.append(iscell_original, iscell_extra)
    unique_cells = np.unique(iscell_combined)
    
    # backup old iscell
    np.save('iscell_backup.npy', iscell)
    
    iscell[unique_cells,0] = 1
    
    # save new iscell
    np.save('iscell.npy', iscell)