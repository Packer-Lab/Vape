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
        
    plane = 0
    obj_list = []

    for pkl_file in pkl_list:
            
        with open(pkl_file, 'rb') as f:
            exp_obj = pickle.load(f)
                
        obj_list = [exp_obj.photostim_r, exp_obj.photostim_s]

        if exp_obj.spont.n_frames > 0:
            obj_list.append(exp_obj.spont)

        if exp_obj.whisker_stim.n_frames > 0:
            obj_list.append(exp_obj.whisker_stim)

        for sub_obj in obj_list:
                
            print('\nMaking STA movie for:', sub_obj.tiff_path)

            size_y = sub_obj.frame_y
            size_x = sub_obj.frame_x
            size_z = (sub_obj.pre_frames*2) + int(sub_obj.stim_dur/sub_obj.fps)
                
            trial_stack = np.empty([0, size_z, size_y, size_x])
                
            for file in os.listdir(sub_obj.tiff_path):
                if '.tif' in file:
                    tiff_file = os.path.join(sub_obj.tiff_path, file)
                    break
                
            for t in range(sub_obj.n_trials):
                frame_start = sub_obj.stim_start_frames[plane][t]
                trial_start = frame_start - sub_obj.pre_frames
                trial_end = frame_start + sub_obj.pre_frames + int(sub_obj.stim_dur/sub_obj.fps)

                if trial_end <= sub_obj.n_frames: # for if the tiff file is incomplete (due to corrupt data)
                    trial = tf.imread(tiff_file, key=range(trial_start, trial_end))
                    trial = np.expand_dims(trial,axis=0)
                    trial_stack = np.append(trial_stack, trial, axis=0)
                        
            trial_avg = np.mean(trial_stack, axis=0)
            avg_baseline = trial_avg[: sub_obj.pre_frames, :, :]
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
                exp_obj = pickle.load(f)

            mean_f = np.mean(exp_obj.photostim_r.raw[0], axis=0)

            count = 0

            for frames in exp_obj.frame_list:
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
            exp_obj = pickle.load(f)
        
        obj_list = [exp_obj.photostim_r, exp_obj.photostim_s]

        if exp_obj.spont.n_frames > 0:
            obj_list.append(exp_obj.spont)

        if exp_obj.whisker_stim.n_frames > 0:
            obj_list.append(exp_obj.whisker_stim)
        
        for sub_obj in obj_list:
            
            tiff_path = sub_obj.tiff_path
            
            file_list = os.listdir(tiff_path)
            for file in file_list:
                if '.tif' in file:
                    tiff_file = os.path.join(tiff_path, file)
  
            total_frames = range(0,sub_obj.n_frames) #get the range of frames for this experiment
            start_frames = total_frames[:1000] 
            end_frames = total_frames[-1000:] 
            
            stack_start = tf.imread(tiff_file, key=start_frames)
            stack_end = tf.imread(tiff_file, key=end_frames)
            
            mean_start = np.mean(stack_start, axis=0)
            mean_end = np.mean(stack_end, axis=0)
            
            output_path = os.path.join(save_path, tiff_path.split('/')[-1])
            
            tf.imsave(output_path + '_mean_start.tif', mean_start.astype('int16'))
            tf.imsave(output_path + '_mean_end.tif', mean_end.astype('int16'))