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
from scipy import stats

# global plotting params
params = {'legend.fontsize': 'x-large',
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
plt.rcParams.update(params)
sns.set()
sns.set_style('white')


def meanError(arr, axis=0, n=False):
    
    if not n:
        n = arr.shape[0]
    
    mean = np.nanmean(arr, axis)
    std = np.nanstd(arr, axis)
    ci = 1.960 * (std/np.sqrt(n)) # 1.960 is z for 95% confidence interval, standard deviation divided by the sqrt of N samples (# cells)
    sem = std/np.sqrt(n)
    
    return mean, std, ci, sem


def points_in_circle_np(radius, x0=0, y0=0, ):
    '''Yields the points in a circle of a defined radius and position
    
    Inputs:
        radius -- radius of circle
        x0     -- x coord of circle centre
        y0     -- y coord of circle centre
        
    Yields:
        x      -- x coord of point n in circle
        y      -- y coord of point n in circle
    '''
    
    x_ = np.arange(x0 - radius - 1, x0 + radius + 1, dtype=int)
    y_ = np.arange(y0 - radius - 1, y0 + radius + 1, dtype=int)
    x, y = np.where((x_[:,np.newaxis] - x0)**2 + (y_ - y0)**2 <= radius**2)
    for x, y in zip(x_[x], y_[y]):
        yield x, y

        
def staMovie(output_dir, pkl_list=False):
    '''Function to construct stimulus-triggered average (STA) movie
    Uses tifffile exclusively and takes up a lot of RAM
    Consider using STAMovieMaker without a GUI
    
    Inputs:
        output_dir -- directory to save movie to
        pkl_list   -- list of pickled objects to obtain metadata for STA movie
    '''
    
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
            
    
def cellFluTime(pkl_list, trial_types='pr ps w none', cell_type=False):
    '''Plots the mean raw fluorescence of all curated cells
    
    Inputs:
        pkl_list -- list of pickled objects with cellular raw fluorescence traces
    '''
    
    fig, ax = plt.subplots(nrows=len(pkl_list), ncols=1, figsize=(10,3*len(pkl_list)), sharey=True, sharex=True)

    for i,pkl in enumerate(pkl_list):
            
        print('Measuring mean cell fluorescence for:', pkl, '              ', end='\r')

        with open(pkl, 'rb') as f:
            ses_obj = pickle.load(f)

        mean_f = np.array([])

        if 'pr' in trial_types:
            
            ssf_pr = ses_obj.photostim_r.stim_start_frames[0]

            for frame in ssf_pr:
                frame_slice = slice(frame, frame+ses_obj.photostim_r.duration_frames+1, 1)
                ses_obj.photostim_r.raw[0][:,frame_slice] = np.nan
            
            if cell_type is 'target':
                pr_targ = ses_obj.photostim_r.targeted_cells
                pr_mean = np.nanmean(ses_obj.photostim_r.raw[0][pr_targ], axis=0)
            else:
                pr_mean = np.nanmean(ses_obj.photostim_r.raw[0], axis=0)

        if 'ps' in trial_types:
            
            ssf_ps = ses_obj.photostim_s.stim_start_frames[0]
            
            for frame in ssf_ps:
                frame_slice = slice(frame, frame+ses_obj.photostim_s.duration_frames+1, 1)
                ses_obj.photostim_s.raw[0][:,frame_slice] = np.nan
                
            if cell_type is 'target':
                ps_targ = ses_obj.photostim_s.targeted_cells
                ps_mean = np.nanmean(ses_obj.photostim_s.raw[0][ps_targ], axis=0)
            else:
                ps_mean = np.nanmean(ses_obj.photostim_s.raw[0], axis=0)

        mean_f = np.concatenate((pr_mean, ps_mean))

        if 'w' in trial_types and ses_obj.whisker_stim.n_frames > 0:
            mean_f = np.concatenate((mean_f, np.mean(ses_obj.whisker_stim.raw[0], axis=0)))

        if 'none' in trial_types and ses_obj.spont.n_frames > 0:
            mean_f = np.concatenate((mean_f, np.mean(ses_obj.spont.raw[0], axis=0)))

        count = 0

        for frames in ses_obj.frame_list:
            x = range(count,count+frames)

            if max(x) < mean_f.shape[0]:
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
        
        
def frameFluTime(data_folder, legend=False):
    '''Plots results from downsampleTiff function
    The grand mean of the first and last 1000 frames from all exps in order
    
    Inputs:
        data_folder -- directory containing tiff stacks
        legend      -- boolean indicating whether to plot the legend
    '''
        
        
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
    '''Saves the mean of the entire first and last 1000 frames of timeseries
    
    Inputs:
        pkl_list  -- list of pickled objects to take metadata from
        save_path -- directory to save mean start/end frames in
    '''
        
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
    '''Return mean image saved by Suite2p
    
    Inputs: 
        s2p_path -- directory with outputs from Suite2p ('save_path0')
    
    Returns:
        mean_img -- 2D uint16 array of the mean image from Suite2p
    '''
    
    os.chdir(s2p_path)
        
    ops = np.load('ops.npy', allow_pickle=True).item()
        
    mean_img = ops['meanImg']

    mean_img = np.array(mean_img, dtype='uint16')
    
    return mean_img
    
    
def s2pMasks(s2p_path, cell_ids):
    '''Return image of cell masks with pixel value corresponding to index
    
    Inputs:
        s2p_path -- directory with outputs from Suite2p ('save_path0')
        cell_ids -- indices of cells to add to the image
    
    Returns:
        mask_img -- 2D uint16 array with cell ROIs filled with cell index value
    '''
    
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
    '''Return image of SLM targets
    
    Inputs:
        obj      -- pickled object containing SLM target areas attribute
        
    Returns:
        targ_img -- 2D uint16 array with SLM target areas filled with 255
    '''
    
    targ_img = np.zeros((obj.frame_y, obj.frame_x), dtype='uint16')
    targ_areas = obj.target_areas
        
    for targ_area in targ_areas:
        for coord in targ_area:
            targ_img[coord[0], coord[1]] = 255
        
    return targ_img

    
def s2pMaskStack(pkl_list, stam_save_path, parent_folder):
    '''Saves a stack of cell mask images for different types of cells
    Also saves stimulus-triggered average images to compare to masks
    
    Inputs:
        pkl_list       -- list of pickled objects to construct mask images from
        stam_save_path -- directory containing stimulus-triggered average images/movies
        parent_folder  -- directory to save cell mask images to
    '''
    
    for pkl in pkl_list:
                
        print('Retrieving s2p masks for:', pkl, '             ', end='\r')
            
        with open(pkl, 'rb') as f:
            ses_obj = pickle.load(f)
        
        pr_obj = ses_obj.photostim_r
        ps_obj = ses_obj.photostim_s
        w_obj = ses_obj.whisker_stim
        
        # list of cell ids to filter s2p masks by
        cell_id_list = [list(range(1,99999)), 
                        pr_obj.cell_id[0],
                        [pr_obj.cell_id[0][i] for i,b in enumerate(pr_obj.cell_s2[0]) if b],
                        [pr_obj.cell_id[0][i] for i,b in enumerate(pr_obj.targeted_cells) if b],
                        [ps_obj.cell_id[0][i] for i,b in enumerate(ps_obj.targeted_cells) if b],
                       ]
        
        # add whisker_stim if it exists for this session
        if w_obj.n_frames > 0:
            cell_id_list.append([w_obj.cell_id[0][i] for i,b in enumerate(w_obj.sta_sig[0]) if b])
            
            for file in os.listdir(stam_save_path):
                if all(s in file for s in ['AvgImage', w_obj.tiff_path.split('/')[-1]]):
                    w_sta_img = tf.imread(os.path.join(stam_save_path, file))
                    w_sta_img = np.expand_dims(w_sta_img, axis=0)

        # empty stack to fill with images
        stack = np.empty((0, pr_obj.frame_y, pr_obj.frame_x), dtype='uint16')
        
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
            if all(s in file for s in ['AvgImage', pr_obj.tiff_path.split('/')[-1]]):
                pr_sta_img = tf.imread(os.path.join(stam_save_path, file))
                pr_sta_img = np.expand_dims(pr_sta_img, axis=0)
            if all(s in file for s in ['AvgImage', ps_obj.tiff_path.split('/')[-1]]):
                ps_sta_img = tf.imread(os.path.join(stam_save_path, file))
                ps_sta_img = np.expand_dims(ps_sta_img, axis=0)
        
        if w_obj.n_frames > 0:
            stack = np.append(stack, w_sta_img, axis=0)
        stack = np.append(stack, pr_sta_img, axis=0)
        stack = np.append(stack, ps_sta_img, axis=0)
        
        # target images
        pr_targ_img = getTargetImage(pr_obj)
        pr_targ_img = np.expand_dims(pr_targ_img, axis=0)
        stack = np.append(stack, pr_targ_img, axis=0)
        
        ps_targ_img = getTargetImage(ps_obj)
        ps_targ_img = np.expand_dims(ps_targ_img, axis=0)        
        stack = np.append(stack, ps_targ_img, axis=0)
        
        # stack is now: mean_img, all_rois, all_cells, s2_cells, pr_cells, ps_cells, 
        # (whisker,) pr_sta_img, ps_sta_img, pr_target_areas, ps_target_areas
        c,y,x = stack.shape
        stack.shape = 1, 1, c, y, x, 1 # dimensions in TZCYXS order
        
        x_pix = pr_obj.pix_sz_x
        y_pix = pr_obj.pix_sz_y
        
        save_path = os.path.join(parent_folder, pkl.split('/')[-1][:-4] + '_s2p_masks.tif')
        
        tf.imwrite(save_path, stack, imagej=True, resolution=(1/y_pix, 1/x_pix), photometric='minisblack')
          
            
def combineIscell(s2p_path, extra_iscell_path):
    '''Combine and save iscell.npy files from Suite2p
    
    Inputs:
        s2p_path          -- directory with outputs from Suite2p ('save_path0')
        extra_iscell_path -- iscell file to be combined with Suite2p output
    '''
    
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
    
    
def topCells(values, cell_ids, amount):
    '''Return top cells based on values
    
    Inputs:
        values       -- 1D float array of values to choose top from
        cell_ids     -- 1D int array of cell indices to consider
        
    Returns:
        top_cell_ids -- 1D int array of cell indexes
    '''
    
    # Sort new filtered array
    sorted_ids = np.argsort(values)

    # Final ten cell indices
    top_ids = sorted_ids[-amount:]
    top_cell_ids = cell_ids[top_ids]
    
    return top_cell_ids  


def bottomCells(values, cell_ids, amount):
    '''Return bottom cells based on values
    
    Inputs:
        values       -- 1D float array of values to choose bottom from
        cell_ids     -- 1D int array of cell indices to consider
        
    Returns:
        top_cell_ids -- 1D int array of cell indexes
    '''
    
    # Sort new filtered array
    sorted_ids = np.argsort(values)

    # Final ten cell indices
    bottom_ids = sorted_ids[:amount]
    bottom_cell_ids = cell_ids[bottom_ids]
    
    return bottom_cell_ids
    
    
def plotCellSTAs(obj, cell_ids, fig_save_path, save=False):
    '''Plot and save stimulus-triggered average calcium traces (dFF)
    
    Inputs:
        obj           -- pickled object containing calcium traces and metadata
        cell_ids      -- 1D array indices of cells to plot calcium traces for
        fig_save_path -- directory to save figures to
        save          -- boolean whether to save the figures or not
    '''
    
    # STA calcium traces
    stas = np.array(obj.stas[0])
    
    trial_std = np.nanstd(obj.all_trials[0][cell_ids], axis=2)
    trial_sem = trial_std/np.sqrt(obj.n_trials)
    
    plt.figure(figsize=(15,10));
    plt.plot(obj.time, stas[cell_ids].T);
    
    for i,cell in enumerate(cell_ids):
        trial_mean = stas[cell]
        plt.fill_between(obj.time, trial_mean + trial_sem[i], trial_mean - trial_sem[i],
                        alpha=0.2, ec='k');
    
    plt.hlines(0, xmin=-2, xmax=2, linestyles='dashed', colors='k')
    plt.legend(cell_ids);
    plt.title(obj.sheet_name + '_' + obj.stim_type + ' top ten largest STAs')
    plt.ylabel('dF/F (baseline-subtracted)')
    plt.xlabel('time (sec)')
    plt.axis(xmin=-1, xmax=2, ymin=-1, ymax=0.2);
    
    if save:
        plt.savefig(os.path.join(fig_save_path,
                                 obj.sheet_name + '_' + obj.stim_type + '_top_traces.png'), bbox_inches = "tight")
        plt.savefig(os.path.join(fig_save_path,
                                 obj.sheet_name + '_' + obj.stim_type + '_top_traces.svg'), bbox_inches = "tight")
        
        
def plotCellMasks(obj, top_ten_cell_ids, stam_save_path, fig_save_path, save=False):
    '''Plot and save ten selected cell masks, stimulus-triggered average
    and mean calcium images (postage stamps)
    
    Inputs:
        obj              -- pickled object containing metadata to plot
        top_ten_cell_ids -- 1D array of ten cell indices to plot
        stam_save_path   -- directory containing stimulus-triggered average images
        fig_save_path    -- directory to save figures to
        save             -- boolean whether to save the figures or not
    '''
    
    tiff_name = obj.tiff_path.split('/')[-1]
    
    for sta_image in os.listdir(stam_save_path):
        if all(s in sta_image for s in ['Image', tiff_name]):
            sta_avg_img = tf.imread(os.path.join(stam_save_path, sta_image)) 
    
    # Cell coords (y,x)
    cell_pos = np.array(obj.cell_med[0])
    top_ten_pos = cell_pos[top_ten_cell_ids]
    
    fig, ax = plt.subplots(nrows=3, ncols=10, figsize=(15,4), sharey=True)
    fig.suptitle(obj.sheet_name + '_' + obj.stim_type + ' corresponding cell raw, STA and mask images')
    
    for a in ax.reshape(-1): a.axis('off')
    
    for i, cell_med in enumerate(top_ten_pos):
        y = int(cell_med[0])
        x = int(cell_med[1])
        cell_id = top_ten_cell_ids[i]
        pxlb = 20 # pixel buffer

        cell_im = np.zeros([40,40]); xmin = ymin = 0; xmax = ymax = 40
        img_y, img_x = obj.mean_imgE[0].shape
        if y-pxlb<0: # this catches if ROI at top/left edge of FOV
            ymin = np.absolute(y-pxlb) 
            yimgmin = 0
        else: yimgmin = y-pxlb
        if x-pxlb<0: 
            xmin = np.absolute(x-pxlb)
            ximgmin = 0
        else: ximgmin = x-pxlb
        if y+pxlb>img_y: # this catches if ROI at bottom/right edge of FOV
            ymax = ymax-(np.absolute(img_y-(y+pxlb))) 
        if x+pxlb>img_x: 
            xmax = xmax-(np.absolute(img_x-(x+pxlb)))
        crop = obj.mean_imgE[0][yimgmin : y+pxlb, ximgmin : x+pxlb]
        cell_im[ymin:ymax, xmin:xmax] = crop
        ax[0,i].imshow(cell_im)
        ax[0,i].set_title(cell_id)

        mask_im = np.zeros([40,40])
        cell_x = obj.cell_x[0][cell_id]-(x-pxlb)
        cell_y = obj.cell_y[0][cell_id]-(y-pxlb)
        mask_im[cell_y, cell_x] = 255
        ax[1,i].imshow(mask_im)

        sta_cell = sta_avg_img[yimgmin : y+pxlb, ximgmin : x+pxlb]
        ax[2,i].imshow(sta_cell, vmin=0, vmax=10)
        
        if save:
            plt.savefig(os.path.join(fig_save_path,
                                 obj.sheet_name + '_' + obj.stim_type + '_top_cells.png'), bbox_inches = "tight")
            plt.savefig(os.path.join(fig_save_path,
                                 obj.sheet_name + '_' + obj.stim_type + '_top_cells.svg'),bbox_inches = "tight")

            
def plotColour(trial_type):
    '''Function for choosing plotting colour based on trial type
    
    Inputs:
        trial_type   -- string indicating the trial type
        
    Returns:
        trial_colour -- plotting colour for this trial type
    '''
    
    trial_types = np.array(['pr', 'ps', 'w', 'none'])
    colours = np.array(['C0', 'C1', 'C3', 'C2'])
    trial_colour = colours[np.where(trial_types==trial_type)][0]
    
    return trial_colour
            
            
def plotCellPositions(obj, cell_ids, fig_save_path, save=False):
    '''Plot and save positions of cells in image coordinate system
    
    Inputs:
        obj           -- pickled object containing metadata to plot
        cell_ids      -- 1D int array of cell indices to plot
        fig_save_path -- directory to save figures to
        save          -- boolean whether to save the figures or not
    '''
    
    # Cell coords (y,x)
    cell_pos = np.array(obj.cell_med[0])
    pos = cell_pos[cell_ids]
    
    plot_colour = plotColour(obj.stim_type)
    
    plt.figure();
    plt.scatter(pos[:,1], pos[:,0], color=plot_colour)
    
    for i, cell in enumerate(cell_ids):
        plt.text(pos[i,1]+20, pos[i,0]+10, str(cell), fontsize=8)
    
    if any(s in obj.stim_type for s in ['pr', 'ps']):
        for target in obj.target_areas:
            med_targ = target[int(len(target)/2)]
            plt.scatter(med_targ[1], med_targ[0], color='k')
        
    plt.axis([0, obj.frame_x, 0, obj.frame_y])
    plt.ylabel('y_coords (pixels)')
    plt.xlabel('x_coords (pixels)')
    plt.title(obj.sheet_name + '_' + obj.stim_type + ' top cell positions')
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.invert_yaxis()
    
    if save:
        plt.savefig(os.path.join(fig_save_path,
                                 obj.sheet_name + '_' + obj.stim_type + '_top_positions.png'), bbox_inches = "tight")
        plt.savefig(os.path.join(fig_save_path,
                                 obj.sheet_name + '_' + obj.stim_type + '_top_positions.svg'), bbox_inches = "tight")


def responseFreqTrial(trial_bool, cells_bool):
    '''Plot the percentage of cells responding on each trial
    
    Inputs:
        trial_bool -- 2D boolean array whether cell responded on trial [cell x trial]
        cells_bool -- 1D boolean array of cells of interest
        
    Returns:
        trial_responses -- 1D array of % of trials each cell responded on
        trial_bins      -- 1D array of range 1:n(trials)
    '''
    
    trial_bool = trial_bool[cells_bool, :]
    
    trial_responses = np.sum(trial_bool, axis=0)/trial_bool.shape[0] * 100
    trial_bins = np.arange(trial_responses.shape[0]);
    
    return trial_responses, trial_bins


def plotResponseFreqTrial(obj, trial_bool, cells_bool, ax):
    '''Plot the percentage of cells responding on each trial
    
    Inputs:
        obj        -- pickled object containing metadata
        trial_bool -- 2D boolean array whether cell responded on trial [cell x trial]
        cells_bool -- 1D boolean array of cells of interest
        ax         -- axis object on which to plot
    '''
    
    trial_responses, trial_bins = responseFreqTrial(trial_bool, cells_bool)
    
    plot_colour = plotColour(obj.stim_type)
    
    ax.hist(trial_bins, weights=trial_responses, bins=trial_bins,
                 align='left', histtype='step',
                 alpha=0.6, lw=2, color=plot_colour, label=obj.stim_type);
    ax.set_xlabel('trial #')
    ax.set_ylabel('% cells responding')
    ax.set_title(obj.sheet_name + ' cell responses over time')
    ax.legend()

    
def plotResponseFreqCell(obj, trial_bool, cells_bool, ax):
    '''Plot the percentage of trials each cell showed a response
    
    Inputs:
        obj        -- pickled object containing metadata
        trial_bool -- 2D boolean array whether cell responded on trial [cell x trial]
        cells_bool -- 1D boolean array of cells of interest
        ax         -- axis object on which to plot
    '''
    
    trial_bool = trial_bool[cells_bool, :]
    
    cell_responses = np.sum(trial_bool, axis=1)/trial_bool.shape[1] * 100
    cell_bins = np.arange(cell_responses.shape[0]);
    
    plot_colour = plotColour(obj.stim_type)
    
    ax.hist(cell_bins, weights=cell_responses, bins=cell_bins,
                 align='left', histtype='step',
                 alpha=0.6, color=plot_colour, label=obj.stim_type);
    ax.set_xlabel('cell #');
    ax.set_ylabel('% trials with responses');
    ax.set_title(obj.sheet_name + ' trial responses per cell');
        
    if any(s in obj.stim_type for s in ['pr', 'ps']):
        targets = np.where(obj.targeted_cells[cells_bool])[0]
        val = 102 if obj.stim_type == 'pr' else 105
        ax.scatter(targets, np.repeat(val, targets.shape),
                   s=10, color=plot_colour, label=obj.stim_type + ' target')
        
    ax.legend()
    

def plotCellResponseRaster(obj, trial_bool, cells_bool, ax):
    '''Plot a raster of trial responses [trial x cell]
    
    Inputs:
        obj        -- pickled object containing metadata
        trial_bool -- 2D boolean array whether cell responded on trial [cell x trial]
        cells_bool -- 1D boolean array of cells of interest
        ax         -- axis object on which to plot [2 x 2]
    '''
    
    trial_responses, trial_bins = responseFreqTrial(trial_bool, cells_bool)
    
    trial_bool = trial_bool[cells_bool, :]
    cell_raster = np.where(trial_bool)
    
    cell_responses =  np.sum(trial_bool, axis=1)/trial_bool.shape[1] * 100
    cell_bins = np.arange(cell_responses.shape[0]);
    
    plot_colour = plotColour(obj.stim_type)
    
    ax[0,0].hist(trial_bins, weights=trial_responses, bins=trial_bins,
                 align='left', histtype='step', orientation='vertical',
                 alpha=0.6, lw=3, color=plot_colour, label=obj.stim_type);
    ax[0,0].set_ylabel('% cells responding');
    ax[0,0].set_title('Cell responses over time');
    ax[0,0].legend();
    
    ax[0,1].set_frame_on(False);

    ax[1,0].scatter(cell_raster[1], cell_raster[0], 
                    s=15, alpha=0.8, color=plot_colour, label=obj.stim_type);
    ax[1,0].set_ylabel('cell #');
    ax[1,0].set_xlabel('trial #');
    ax[1,0].set_title(obj.sheet_name + ' raster of single trial responses');

    ax[1,1].hist(cell_bins, weights=cell_responses, bins=cell_bins, 
                 align='left', histtype='step', orientation='horizontal',
                 alpha=0.6, color=plot_colour, label=obj.stim_type);
    ax[1,1].set_xlabel('% trials responded on');
    ax[1,1].yaxis.set_label_position("right");
    ax[1,1].set_ylabel('Trial responses per cell', rotation=270);
    
    if any(s in obj.stim_type for s in ['pr', 'ps']):
        targets = np.where(obj.targeted_cells[cells_bool])[0]
        val = 102 if obj.stim_type == 'pr' else 105
        ax[1,1].scatter(np.repeat(val, targets.shape), targets,
                   s=10, color=plot_colour, label=obj.stim_type + ' target')
    ax[1,1].legend(loc='upper center')
    

def responseAmpTrial(obj, trial_bool, cells_bool):
    '''Plot the percentage of cells responding on each trial
    
    Inputs:
        obj        -- pickled object containing metadata
        trial_bool -- 2D boolean array whether cell responded on trial [cell x trial]
        cells_bool -- 1D boolean array of cells of interest
        ax         -- axis object on which to plot
        
    Returns: 
        trial_amp_means -- mean of all cell dFF changes per trial
    '''
    
    trial_bool = trial_bool[cells_bool, :]
    
    resp_cell_amps = copy.deepcopy(obj.all_amplitudes[0][cells_bool]) # amplitudes of responders [cell x trial]
    resp_cell_amps[~trial_bool] = np.nan

    trial_amp_means = np.nanmean(resp_cell_amps, axis=0)
    
    return trial_amp_means
    
    
def plotResponseAmpTrial(obj, trial_bool, cells_bool, ax):
    '''Plot the percentage of cells responding on each trial
    
    Inputs:
        obj        -- pickled object containing metadata
        trial_bool -- 2D boolean array whether cell responded on trial [cell x trial]
        cells_bool -- 1D boolean array of cells of interest
        ax         -- axis object on which to plot
    '''

    trial_amp_means = responseAmpTrial(obj, trial_bool, cells_bool)
    
    plot_colour = plotColour(obj.stim_type)
    
    ax.plot(trial_amp_means, alpha=0.8, lw=2, color=plot_colour, label=obj.stim_type);
    ax.set_xlabel('trial #')
    ax.set_ylabel('Change in dFF')
    ax.set_title(obj.sheet_name + ' mean cell change in dFF per trial')
    ax.axis([-0.5, 101, 0, 3.5])
    ax.legend(loc='upper right')    
    
    
def responseAmpSumTrial(obj, trial_bool, cells_bool):
    '''Plot the percentage of cells responding on each trial
    
    Inputs:
        obj        -- pickled object containing metadata
        trial_bool -- 2D boolean array whether cell responded on trial [cell x trial]
        cells_bool -- 1D boolean array of cells of interest
        ax         -- axis object on which to plot
        
    Returns:
        trial_amp_sum -- sum of all dFF changes per trial
    '''
    
    trial_bool = trial_bool[cells_bool, :]
    
    resp_cell_amps = copy.deepcopy(obj.all_amplitudes[0][cells_bool]) # amplitudes of responders [cell x trial]
    resp_cell_amps[~trial_bool] = np.nan

    trial_amp_sum = np.nansum(resp_cell_amps, axis=0) 
    
    return trial_amp_sum


def plotResponseAmpSumTrial(obj, trial_bool, cells_bool, ax):
    '''Plot the percentage of cells responding on each trial
    
    Inputs:
        obj        -- pickled object containing metadata
        trial_bool -- 2D boolean array whether cell responded on trial [cell x trial]
        cells_bool -- 1D boolean array of cells of interest
        ax         -- axis object on which to plot
    '''
    
    trial_amp_sum = responseAmpSumTrial(obj, trial_bool, cells_bool)
        
    plot_colour = plotColour(obj.stim_type)
    
    ax.plot(trial_amp_sum, alpha=0.8, lw=2, color=plot_colour, label=obj.stim_type);
    ax.set_xlabel('trial #')
    ax.set_ylabel('Summed change in dFF')
    ax.set_title(obj.sheet_name + ' mean summed change in dFF per trial')
    ax.legend(loc='upper right')  

    
def listdirFullpath(directory, string=''):
    '''Return full path of all files in directory containing specified string
    
    Inputs:
        directory -- path to directory (string)
        string    -- sequence to be found in file name (string)
    '''
    return [os.path.join(directory, file) \
                for file in os.listdir(directory) \
                    if string in file]


def loadPickle(pickle_path):
    '''Load the pickled object
    
    Inputs:
        pickle_path -- path to pickled object
    '''
    
    print('Loading pickle:', pickle_path)

    with open(pickle_path, 'rb') as f:
        obj = pickle.load(f)
    
    return obj


def makeExpList(ses_obj, stim_types):
    '''Construct experiment list from experiments in
    object
    
    Inputs:
        ses_obj -- session object created using interareal_analysis.py
    '''
    
    exp_list = []
    
    if ses_obj.photostim_r.n_frames > 0 and 'pr' in stim_types: 
        exp_list.append(ses_obj.photostim_r)
    if ses_obj.photostim_s.n_frames > 0 and 'ps' in stim_types: 
        exp_list.append(ses_obj.photostim_s)
    if ses_obj.spont.n_frames > 0 and 'none' in stim_types: 
        exp_list.append(ses_obj.spont)            
    if ses_obj.whisker_stim.n_frames > 0 and 'w' in stim_types: 
        exp_list.append(ses_obj.whisker_stim)
        
    return exp_list
    
    
def getMaxTrialLength(ses_obj_list):
    '''Find the maximum trial length for this series of experiments
    
    Inputs:
        obj_list     -- list of session objects made using interareal_analysis.py
        
    Outputs:
        trial_len_max -- maximum trial length from all stim types
    '''
    trial_len_max = 0
    
    for trial_type in ses_obj_list:
        trial_len = len(trial_type.time)
        
        if trial_len > trial_len_max:
            trial_len_max = trial_len
            
    return trial_len_max


def getResponderIdentities(exp_obj, sign_filter, sig='fdr'):
    '''Get the number of responding cells according to statistical
    threshold for targets, s1 non-target and s2
    
    Inputs:
        obj           -- experiment object created using interareal_analysis.py
        sign_filter   -- whether the amplitude of response for each 
                         cell matches the filter (bool)
        sig           -- the method of calculating significant responders
    Ouput:
        []            -- list of whether cells responded or not
    '''
    if sig=='fdr': sig_filter = exp_obj.sta_sig[0]
    if sig=='nomulti': sig_filter = exp_obj.sta_sig_nomulti[0]
    if sig=='insig-fdr': sig_filter = ~exp_obj.sta_sig[0]
    if sig=='insig-nomulti': sig_filter = ~exp_obj.sta_sig_nomulti[0]
        
    s1_target_responders = exp_obj.targeted_cells & sig_filter
    s1_nontarget_responders = exp_obj.cell_s1[0] & ~exp_obj.targeted_cells & sig_filter & sign_filter
    s2_responders = exp_obj.cell_s2[0] & sig_filter & sign_filter
    
    return [s1_target_responders, s1_nontarget_responders, s2_responders]


def getResponderTrials(exp_obj, responders):
    '''Get the mean trials for each cell type
    
    Inputs:
        obj        -- object created using interareal_analysis.py
        responders -- whether cells were responding (list of bools)
    '''
    
    all_trials = exp_obj.all_trials[0]
    
    s1_targ_trials = np.nanmean(all_trials[responders[0]], axis=(0,2)) # mean across cell and trial
    s1_nt_trials = np.nanmean(all_trials[responders[1]], axis=(0,2))
    s2_trials = np.nanmean(all_trials[responders[2]], axis=(0,2))
    
    return [s1_targ_trials, s1_nt_trials, s2_trials]


def fillList(list_, var, i_0=0):
    '''Fill array from the beginning without providing all indices'''
    
    if len(np.shape(var)) > 1 or len(np.shape(list_)) > 1:
        raise Exception('inputs must be lists')
    
    i_n = len(var)
    
    if i_n+i_0 > len(list_):
        raise Exception('list is too long to be inserted')
        
    list_[i_0:i_n] = var
           
    return list_


def filterDfBoolCol(df, true_cols=[], false_cols=[]):
    '''Filter indices in a pandas dataframe using logical operations
    on columns with Boolean values
    
    Inputs:
        df         -- dataframe
        true_cols  -- columns where True should be filtered
        false_cols -- columns where False should be filtered
    
    Outputs:
        indices of the dataframe where the logical operation is true
    '''
    if true_cols: 
        true_rows = df[true_cols].all(axis='columns')
    
    if false_cols:
        false_rows = (~df[false_cols]).all(axis='columns')
    
    if true_cols and false_cols:
        filtered_df = df[true_rows & false_rows]
    elif true_cols:
        filtered_df = df[true_rows]
    elif false_cols:
        filtered_df = df[false_rows]
    
    return filtered_df.index


def savePlot(save_path):
    '''Save both .png and .svg from a matplotlib plot
    
    Inputs:
        save_path -- path to save plots to
    '''
    plt.savefig(save_path + '.png', bbox_inches='tight')
    plt.savefig(save_path + '.svg', bbox_inches='tight')
    
    
def stat_test_timepoint(df, time_array, col_1='', col_2='', frames_bin=2, th=0.05):
    '''
    time array with time in seconds, should be same size as accuracy arrays 
    use nans to exclude (artefact) periods
    '''
    
    # df with spont, pr, ps in across time array
    signif_array = np.zeros(len(time_array))
    n_bins = int(np.floor(np.sum(~np.isnan(time_array)) / frames_bin))  # exclude artefact in test
    th_bonf = th / n_bins  # perform bonferroni correction for number of tests

    for i_bin in range(n_bins):  # loop through bins
        start_frame = int(i_bin * frames_bin)
        end_frame = int((i_bin + 1) * frames_bin)
        time_min = time_array[start_frame]
        if end_frame >= len(time_array):
            time_max = time_array[-1] + 0.1
            end_frame = len(time_array) 
        else:
            time_max = time_array[end_frame]
        
        if np.sum(np.isnan(time_array[start_frame:end_frame + 1])) > 0:
            continue  # skip bins that contains nans [during artefact]
        else:
            inds_rows = np.logical_and(df['timepoint'] >= time_min, 
                                    df['timepoint'] < time_max)
            sub_df = df[inds_rows]  # select df during this time bin

            stat, pval = stats.wilcoxon(x=sub_df[col_1], y=sub_df[col_2], 
                                            alternative='two-sided')
            if pval < th_bonf:
                signif_array[start_frame:end_frame] = 1  # indicate significance

    return signif_array