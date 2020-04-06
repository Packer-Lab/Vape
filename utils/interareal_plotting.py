import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd

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

class interarealPlotting():

    def __init__(self, pkl_folder):
        
        self.pkl_folder = pkl_folder
        self.pkl_name = []
        self.pkl_path = []

        self.n_units = []
        self.n_targets = []
        self.n_targeted_cells = []
        self.targeted_cells = []
        self.s1_cells = []
        self.stim_dur = []
        self.stim_freq = []
        self.stim_type = []
        self.sheet_name = []
        self.tiff_path = []
        self.fps = []

        self.df = pd.DataFrame()

        self.addPickles()

    def _parseSimpleMetadata(self, sub_obj):
              
        self.stim_type.append(sub_obj.stim_type)
        self.tiff_path.append(os.path.split(sub_obj.tiff_path)[1])
        self.fps.append(sub_obj.fps)
        self.n_units.append(sub_obj.n_units[0])
        self.s1_cells.append(np.array(sub_obj.cell_s1[0]))
        
        df = pd.DataFrame({'sheet_name'       : [self.sheet_name[-1]],
                           'tiff_path'        : [self.tiff_path[-1]],
                           'stim_type'        : [self.stim_type[-1]],
                           'fps'              : [self.fps[-1]],
                           'n_units'          : [self.n_units[-1]], 
                           's1_cells'         : [self.s1_cells[-1]]
                          }
                         )
        
        self.temp_df = pd.concat([self.temp_df, df], axis=1, sort=False)
        
    def _meanSTA(self, sub_obj):
        
        s1_sta_amp = []
        s1_sta = []
        s2_sta_amp = []
        s2_sta = []
        
        for cell,_ in enumerate(sub_obj.cell_id[0]):
            if sub_obj.cell_s1[0][cell] == 0:
                s2_sta.append(sub_obj.stas[0][cell])
                s2_sta_amp.append(sub_obj.sta_amplitudes[0][cell])
            if sub_obj.cell_s1[0][cell] == 1:
                s1_sta.append(sub_obj.stas[0][cell])
                s1_sta_amp.append(sub_obj.sta_amplitudes[0][cell])
    
        df = pd.DataFrame({'s2_sta' : [np.nanmean(s2_sta,axis=0)],
                           's2_sta_amp' : [np.nanmean(s2_sta_amp,axis=0)],
                           's2_sta_std' : [np.std(s2_sta, axis=0)],
                           's1_sta' : [np.nanmean(s1_sta,axis=0)],
                           's1_sta_amp' : [np.nanmean(s1_sta_amp,axis=0)],
                           's1_sta_std' : [np.std(s1_sta, axis=0)]
                          }
                         )

        self.temp_df = pd.concat([self.temp_df, df], axis=1, sort=False)
    
    def _numCellsRespond(self, sub_obj):

        #number of cell in s1 and s2 based on s2p ROIs in certain parts of the image
        num_s1_cells = sub_obj.cell_s1[0].count(True)
        num_s2_cells = sub_obj.cell_s1[0].count(False)

        #amplitudes of response using stimulus triggered average dff pre/post stim
        amps = sub_obj.all_amplitudes[0]
        pos_amps = (amps > 0).T
        neg_amps = (amps <= 0).T
        
        #significant single trials for each cell (response >2 S.D. of the baseline)
        single_sig = (np.array(sub_obj.single_sig[0])).T
        
        #boolean of which cells are in s1 or s2
        s1_cells = np.array(sub_obj.cell_s1[0]) # boolean of length cell for s1 cells
        s2_cells = ~s1_cells

        #positive, negative or all responding cells in s1 or s2 for each trial
        pos_s1 = pos_amps & single_sig & s1_cells
        pos_s2 = pos_amps & single_sig & s2_cells
        neg_s1 = neg_amps & single_sig & s1_cells 
        neg_s2 = neg_amps & single_sig & s2_cells
        
        df = pd.DataFrame({'num_s1_cells' : [num_s1_cells],
                           'num_s2_cells' : [num_s2_cells],
                           'positive_s1_responders_trial' : [np.sum(pos_s1, axis=1)],
                           'negative_s1_responders_trial' : [np.sum(neg_s1, axis=1)],
                           'positive_s2_responders_trial' : [np.sum(pos_s2, axis=1)],
                           'negative_s2_responders_trial' : [np.sum(neg_s2, axis=1)],
                          }  
                         )

        self.temp_df = pd.concat([self.temp_df, df], axis=1, sort=False)
        
        #amplitudes of response using stimulus triggered average dff pre/post stim
        amps = sub_obj.sta_amplitudes[0]
        pos_amps = (amps > 0).T
        neg_amps = (amps <= 0).T
        
        #boolean of reliable responders (significant t-test between 100 pairs of pre and post mean dffs)
        sta_sig = np.array(sub_obj.sta_sig[0])
        sta_sig_nomulti = np.array(sub_obj.sta_sig_nomulti[0])
        
        #cells responding sta
        sta_sig_s1_pos = sta_sig & s1_cells & pos_amps
        sta_sig_s2_pos = sta_sig & s2_cells & pos_amps
        sta_sig_nomulti_s1_pos = sta_sig_nomulti & s1_cells & pos_amps
        sta_sig_nomulti_s2_pos = sta_sig_nomulti & s2_cells & pos_amps
        sta_sig_s1_neg = sta_sig & s1_cells & neg_amps
        sta_sig_s2_neg = sta_sig & s2_cells & neg_amps
        sta_sig_nomulti_s1_neg = sta_sig_nomulti & s1_cells & neg_amps
        sta_sig_nomulti_s2_neg = sta_sig_nomulti & s2_cells & neg_amps
        
        df = pd.DataFrame({'positive_s1_responders_sta' : [np.sum(sta_sig_s1_pos)],
                           'negative_s1_responders_sta' : [np.sum(sta_sig_s1_neg)],
                           'positive_s2_responders_sta' : [np.sum(sta_sig_s2_pos)],
                           'negative_s2_responders_sta' : [np.sum(sta_sig_s2_neg)],
                           'positive_s1_responders_sta_nomulti' : [np.sum(sta_sig_nomulti_s1_pos)],
                           'negative_s1_responders_sta_nomulti' : [np.sum(sta_sig_nomulti_s1_neg)],
                           'positive_s2_responders_sta_nomulti' : [np.sum(sta_sig_nomulti_s2_pos)],
                           'negative_s2_responders_sta_nomulti' : [np.sum(sta_sig_nomulti_s2_neg)]
                          }  
                         )

        self.temp_df = pd.concat([self.temp_df, df], axis=1, sort=False)
    
    def _probabilityResponse(self, sub_obj):

        # For each sub_obj, i.e. photostim_r, photostim_s etc.
        # calculate for each cell the probability of response and save to dataframe?

        # single_sig is [plane][cell][trial]
        n_trials = sub_obj.n_trials

        # get the number of responses
        num_respond = np.array(sub_obj.single_sig[0])

        # get the probability of response
        prob_response = np.sum(num_respond, axis=1) / n_trials

        df = pd.DataFrame({'prob_response' : [prob_response]
                        }  
                        )

        self.temp_df = pd.concat([self.temp_df, df], axis=1, sort=False)   
    
    def _parsePhotostimMetadata(self, sub_obj):
        
        self.n_targets.append(sub_obj.n_targets)
        self.targeted_cells.append(np.array(sub_obj.targeted_cells) > 0)
        self.n_targeted_cells.append(len([i for i in sub_obj.targeted_cells if i==1]))
        self.stim_dur.append(sub_obj.stim_dur)
        self.stim_freq.append( ( 1 / ( ( (sub_obj.single_stim_dur*sub_obj.n_shots) * sub_obj.n_groups-1 ) + ( sub_obj.inter_point_delay * sub_obj.n_groups ) ) ) *1000 )
        
        df = pd.DataFrame({'n_targets'        : [self.n_targets[-1]], 
                           'target_cells'     : [self.targeted_cells[-1]],
                           'n_targeted_cells' : [self.n_targeted_cells[-1]],
                           'stim_dur'         : [self.stim_dur[-1]],
                           'stim_freq'        : [self.stim_freq[-1]]
                          }
                         )

        self.temp_df = pd.concat([self.temp_df, df], axis=1, sort=False)
    
    def _meanTargetSTA(self, sub_obj):
        
        targeted_sta_amp = []
        targeted_sta = []
        non_targeted_sta_amp = []
        non_targeted_sta = []
        
        for cell,_ in enumerate(sub_obj.targeted_cells):
            if sub_obj.targeted_cells[cell] == 1:
                targeted_sta.append(sub_obj.stas[0][cell])
                targeted_sta_amp.append(sub_obj.sta_amplitudes[0][cell])
            else:
                non_targeted_sta.append(sub_obj.stas[0][cell])
                non_targeted_sta_amp.append(sub_obj.sta_amplitudes[0][cell])
    
        df = pd.DataFrame({'target_sta' : [np.nanmean(targeted_sta,axis=0)],
                           'target_sta_amp' : [np.nanmean(targeted_sta_amp,axis=0)],
                           'target_sta_std' : [np.std(targeted_sta, axis=0)],
                           'non_target_sta' : [np.nanmean(non_targeted_sta,axis=0)],
                           'non_target_sta_amp' : [np.nanmean(non_targeted_sta_amp,axis=0)],
                           'non_target_sta_std' : [np.std(non_targeted_sta, axis=0)]
                          }
                         )

        self.temp_df = pd.concat([self.temp_df, df], axis=1, sort=False)
        
    def _numTargetsRespond(self, sub_obj):
        
        #number of cell in s1 and s2 based on s2p ROIs in certain parts of the image
        num_s1_cells = sub_obj.cell_s1[0].count(True)
        num_s2_cells = sub_obj.cell_s1[0].count(False)

        #amplitudes of response using stimulus triggered average dff pre/post stim
        amps = sub_obj.all_amplitudes[0]
        pos_amps = (amps > 0).T
        neg_amps = (amps <= 0).T

        #significant single trials for each cell (response >2 S.D. of the baseline)
        single_sig = (np.array(sub_obj.single_sig[0])).T
        
        #boolean of targeted cells
        target_cells = np.array(sub_obj.targeted_cells)
        targeted_cells = target_cells > 0
        
        sig_targeted = targeted_cells & pos_amps & single_sig 
                
        df = pd.DataFrame({'target_responders_trial' : [sig_targeted],
                           'target_responders_trial_sum' : [np.sum(sig_targeted, axis=1)]
                          }  
                         )
        
        self.temp_df = pd.concat([self.temp_df, df], axis=1, sort=False)
        
        #amplitudes of response using stimulus triggered average dff pre/post stim
        amps = sub_obj.sta_amplitudes[0]
        pos_amps = (amps > 0).T
        
        #boolean of reliable responders (significant t-test between 100 pairs of pre and post mean dffs)
        sta_sig = np.array(sub_obj.sta_sig[0])
        sta_sig_nomulti = np.array(sub_obj.sta_sig_nomulti[0])
        
        sta_sig_target = sta_sig & targeted_cells & pos_amps
        sta_sig_nomulti_target = sta_sig_nomulti & targeted_cells & pos_amps
        
        df = pd.DataFrame({'target_responders' : [sta_sig_target],
                           'target_responders_sta' : [np.sum(sta_sig_target)],
                           'target_responders_sta_nomulti' : [np.sum(sta_sig_nomulti_target)]
                          }  
                         )
        
        self.temp_df = pd.concat([self.temp_df, df], axis=1, sort=False)
    
    def _stimTrialParameters(self, sub_obj, whisker_cells):
        
        df = self.df

        # sheet name for current session
        sheet_name = sub_obj.sheet_name
        stim_type = sub_obj.stim_type
        targeted_cells = np.where(sub_obj.targeted_cells)[0]
        all_responses = np.array(sub_obj.all_amplitudes[0])

        # preallocation of list for collection of amplitudes later
        sum_dff_trials = []
        num_whisker_targets = []
        dists = []
        
        n_trials = np.shape(sub_obj.all_trials)[2]

        for trial in range(n_trials):
            responders = [i for i in range(sub_obj.n_units[0]) if sub_obj.single_sig[0][i][trial] == 1]
            target_responder_ids = [i for i in responders if i in targeted_cells]
            
            if whisker_cells:
                num_whisker_targets.append(sum(1 for i in whisker_cells[0] if i in target_responder_ids))

            target_responses = all_responses[target_responder_ids] # responses of only the targets in dFF
            sum_dff = np.sum(target_responses[:,trial], axis=0) # sum of those responses
            sum_dff_trials = np.append(sum_dff_trials, sum_dff) # append to list of all trials summed dFF
            
            cell_positions = np.array(sub_obj.cell_med[0])

            resp_positions = cell_positions[target_responder_ids]

            if np.any(resp_positions):
                targ_coords = list(zip(*resp_positions))
                centroidx = np.sum(targ_coords[0])/len(targ_coords[0])
                centroidy = np.sum(targ_coords[1])/len(targ_coords[1])
                centroid = [centroidx, centroidy]

                targ = resp_positions[0]
                dist = np.linalg.norm(targ-centroid)
                dists.append(dist)
            else: 
                dists.append(0.0)
                
        temp_df = pd.DataFrame({'target_sum_dff' : [sum_dff_trials],
                                'num_whisker_targets' : [num_whisker_targets],
                                'euclid_dist' : [dists]
                                  }  
                                 )

        # save the results to the df
        self.temp_df = pd.concat([self.temp_df, temp_df], axis=1, sort=False)
                
    def _performAnalysis(self):

        for pkl_file in self.new_pkls:
            print('Collecting analysed data for pickled object:', pkl_file, '          ', end='\r')
            whisker_cells = False
            
            basename = os.path.basename(pkl_file)
            self.pkl_name.append(basename)

            with open(pkl_file, 'rb') as f:
                exp_obj = pickle.load(f)
            
            pkl_list = [exp_obj.photostim_r, exp_obj.photostim_s]

            if exp_obj.spont.n_frames > 0:
                pkl_list.append(exp_obj.spont)
                
            if exp_obj.whisker_stim.n_frames > 0:
                pkl_list.append(exp_obj.whisker_stim)
                whisker_cells = np.where(exp_obj.whisker_stim.sta_sig[0]) # find the number of whisker responsive cells targeted on each trial

            for sub_obj in pkl_list:
                
                self.temp_df = pd.DataFrame()

                self.sheet_name.append(exp_obj.sheet_name)

                self._parseSimpleMetadata(sub_obj)
                
                self._meanSTA(sub_obj)

                self._numCellsRespond(sub_obj)
                
                self._probabilityResponse(sub_obj)
                
                if any(s in sub_obj.stim_type for s in ['pr', 'ps', 'none']):
                    
                    self._parsePhotostimMetadata(sub_obj)

                    self._meanTargetSTA(sub_obj)

                    self._numTargetsRespond(sub_obj)

                    self._stimTrialParameters(sub_obj, whisker_cells)
                    
                self.df = self.df.append(self.temp_df, ignore_index=True, sort=False)

    def addPickles(self):

        pkl_folder = self.pkl_folder
        self.new_pkls = []

        for file in os.listdir(pkl_folder):
            if '.pkl' in file and file not in self.pkl_name:
                path = os.path.join(pkl_folder, file)
                self.pkl_path.append(path)

                self.new_pkls.append(path)
        
        self._performAnalysis()

    def boxplotSummaryStat(self, column):

        print('Plotting summary statistic for all experiments:', column)
        plt.figure()

        if type(column) is not str:
            raise Exception('ERROR: column variable is not a string')

        df = self.df

        sns.boxplot(x='stim_type', y=column, data=df, width=0.2)
        sns.swarmplot(x='stim_type', y=column, data=df, color='k', size=5)
                
    def distributionSummaryStat(self, column):

        print('Plotting summary statistic for all experiments:', column)
        plt.figure()

        if type(column) is not str:
            raise Exception('ERROR: column variable is not a string')

        df = self.df

        grouped = df.groupby('stim_type')

        fig, ax = plt.subplots(nrows=len(grouped), ncols=1, figsize=(15,15), sharey=True, sharex=True)
        plot_index = 0
        
        labels = []
        
        for name, group in grouped:
            for i, row in group.iterrows():
                x = row.loc[column]
                ax[plot_index] = sns.distplot(x, rug=True, hist=False, ax=ax[plot_index])
                ax[plot_index].set_title(name)
                ax[plot_index].set_xlim(-100, 100)
                labels.append(row.loc['sheet_name'])
            
            ax[plot_index].legend(labels) 
            ax[plot_index].set_xlabel(column)
            ax[plot_index].set_ylabel('Relative frequency')
            plot_index += 1

    def lineplotSTA(self, column):

        print('Plotting stimulus triggered average for all experiments:', column)

        if type(column) is not str:
            raise Exception('ERROR: column variable is not a string')

        df = self.df

        grouped = df.groupby('stim_type')

        fig, ax = plt.subplots(nrows=1, ncols=len(grouped), figsize=(15,5), sharey=True)
        plot_index = 0

        for _, group in grouped:
            for i, row in group.iterrows():
                y = row.loc[column]
                
                if not np.any(np.isnan(y)):
                    x = list(range(len(y)))
                    x = np.array(x)/row.loc['fps']
                    ax[plot_index].plot(x, y, label=row.loc['sheet_name'])
            plot_index += 1
            
        plt.legend()

    def scatterResponseTrial(self, columns, proportion=False):

        if len(columns) is not 2:
            raise Exception('ERROR: only provide two column strings')
        
        df = self.df.copy(deep=True)

        if proportion:
            df[columns[0]] = df[columns[0]]/df.num_s1_cells 
            df[columns[1]] = df[columns[1]]/df.num_s2_cells
            print('Plotting scatter and slopes for S1 vs S2 proportions for all experiments:', columns)
        else:
            print('Plotting scatter and slopes for S1 vs S2 for all experiments:', columns)

        # Plot all trials of each experiment grouped by stim type
        grouped = df.groupby('stim_type', sort=False)

        fig, ax = plt.subplots(nrows=1, ncols=len(grouped), figsize=(15,5), sharey=True, sharex=True)

        plot_index = 0

        for name, group in grouped:
                
            for i, row in group.iterrows():
                
                # Plot scatter of S1 vs. S2
                x = row.loc[columns[0]]
                y = row.loc[columns[1]]

                if isinstance(x, np.ndarray) or isinstance(x, list): 
                    ax[plot_index].scatter(x, y, label=row.loc['sheet_name'])
                    ax[plot_index].set_title(name)

                    # Plot slope
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    ax[plot_index].plot(x,p(x))
    
            plot_index += 1

        # Get slope for each experiment and each stim_type
        grouped = df.sort_values(['stim_type'], ascending=False).groupby('sheet_name')

        plot_index = 0
        all_slopes = []
        
        labels = []
        
        for name, group in grouped:
            
            slopes = []  
            
            for i, row in group.iterrows():
                x = row.loc[columns[0]]
                y = row.loc[columns[1]]
                
                if isinstance(x, np.ndarray):                   
                    slope, _, r_value, p_value, _ = stats.linregress(x,y)
                    slopes.append(slope)
            
            plot_index += 1
            
            all_slopes.append(slopes)
            labels.append(name)

        plt.figure()

        for slopes in all_slopes:
            x = range(len(slopes))
            y = slopes
            plt.plot(x, y)

        plt.xticks(np.arange(4), ('ps', 'pr', 'none', 'w'))
        plt.legend(labels)

    def scatterResponseSTA(self, columns, proportion=False):

        if len(columns) is not 2:
            raise Exception('ERROR: only provide two column strings')
        
        df = self.df.copy(deep=True)

        if proportion:
            df[columns[0]] = df[columns[0]]/df.num_s1_cells 
            df[columns[1]] = df[columns[1]]/df.num_s2_cells
            print('Plotting reliable responders (STA) for S1 vs S2 proportions:', columns)
        else:
            print('Plotting reliable responders (STA) for S1 vs S2:', columns)

        # Plot all trials of each experiment grouped by stim type
        grouped = df.groupby('stim_type', sort=False)

        fig, ax = plt.subplots(nrows=1, ncols=len(grouped), figsize=(15,5), sharey=True, sharex=True)

        plot_index = 0
        plot_colour = ['darkblue', 'darkorange', 'darkgreen', 'darkred']

        for name, group in grouped:

            # Plot scatter of S1 vs. S2
            x = group.loc[:,columns[0]]
            y = group.loc[:,columns[1]]
            ax[plot_index].scatter(x, y, c=plot_colour[plot_index])
            ax[plot_index].set_title(name)
    
            plot_index += 1

    def boxplotProbResponse(self, to_mask=None):

        df = self.df
        
        # Plot all trials of each experiment grouped by stim type
        grouped = df.groupby('sheet_name', sort=False)
        
        fig, ax = plt.subplots(nrows=len(grouped), ncols=1, figsize=(5,50), sharey=True, sharex=True)

        plot_index = 0
        for name, group in grouped:
            
            if to_mask:
                mask = np.array(group.loc[:,to_mask])[0]
            
            # Get probability of response for each cell of each stim type
            x = np.array(group.loc[:,'prob_response'])
            x = np.concatenate(x, axis=0).reshape(len(group),-1)
            
            if to_mask:
                x = list(x[:,~mask])
            else:
                x = list(x)

            ax[plot_index].boxplot(x)
            ax[plot_index].set_title(name)

            ax[plot_index].set_xticklabels(('pr', 'ps', 'none', 'w'))

            plot_index += 1

    def scatterProbResponse(self, to_mask=None):
        
        df = self.df

        grouped = df.groupby('sheet_name', sort=False)

        for name, group in grouped:

            fig, ax = plt.subplots(nrows=1, ncols=len(group), figsize=(15,5), sharey=True, sharex=True)
            
            if to_mask:
                mask = np.array(group.loc[:,to_mask])[0]

            # Get probability of response for each cell of each stim type
            x = np.array(group.loc[:,'prob_response']) # order of the group = pr, ps, none
            x = np.concatenate(x, axis=0).reshape(len(group),-1)
            if to_mask:
                x = list(x[:,~mask])
            else:
                x = list(x)

            ax[0].scatter(x[0],x[1], c='r', alpha=0.3)
            ax[0].set_title('random vs similar')
            ax[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
            
            # Plot slope
            z = np.polyfit(x[0], x[1], 1)
            p = np.poly1d(z)
            ax[0].plot(x[0],p(x[0]), 'k--', alpha=0.3)
            ax[0].axis([-.01, 0.5, -.01, 0.5])

            if len(group) == 3:
                ax[1].scatter(x[2],x[0], alpha=0.3)
                ax[1].set_title('sham vs random')
                ax[1].plot([0, 1], [0, 1], 'k--', alpha=0.3)
                ax[0].axis([-.01, 0.5, -.01, 0.5])

                # Plot slope
                z = np.polyfit(x[2], x[1], 1)
                p = np.poly1d(z)
                ax[1].plot(x[2],p(x[2]), 'k--', alpha=0.3)

                ax[2].scatter(x[2],x[1], c='orange', alpha=0.3)
                ax[2].set_title('sham vs similar')
                ax[2].plot([0, 1], [0, 1], 'k--', alpha=0.3)
                ax[0].axis([-0.01, 0.5, -0.01, 0.5])

                # Plot slope
                z = np.polyfit(x[2], x[1], 1)
                p = np.poly1d(z)
                ax[2].plot(x[2],p(x[2]), 'k--', alpha=0.3)
    
    def boxplotWhiskerBias(self):
        
        print('\nNumber of cells responding out of total targeted:')
        #get dataframe
        df = self.df

        #preallocate variables
        ps_target_whisker_response = []
        pr_target_whisker_response = []

        #group the dataframe by experiment
        all_groups = df.groupby('sheet_name', sort=False)

        #filter the groups to only those containing the 'stim_type': 'w'
        filtered_df = all_groups.filter(lambda x: (x['stim_type'] == 'w').any())

        #re-group the dataframe that has been filtered
        all_whisker_groups = filtered_df.sort_values(['stim_type']).groupby('sheet_name', sort=True)

        #iterate through the groups
        for name, group in all_whisker_groups:

            #find the targets responding on each trial (boolean list)
            ps_target_responders = np.array(group.loc[group['stim_type'] == 'ps', 'target_responders'])[0]
            pr_target_responders = np.array(group.loc[group['stim_type'] == 'pr', 'target_responders'])[0]  

            #print the number of targets responding out of total number of targeted cells
            print(name)
            print('Similar:', list(ps_target_responders).count(True), 'out of', int(group.loc[group['stim_type'] == 'ps', 'n_targeted_cells']))
            print('Random:', list(pr_target_responders).count(True), 'out of', int(group.loc[group['stim_type'] == 'pr', 'n_targeted_cells']))

            #find the probability of response for all cells to whisker stim
            whisker_prob_response = np.array(group.loc[group['stim_type'] == 'w', 'prob_response'])[0]

            #append the probability of response to whisker stim for the responding target cells (STA)  
            ps_target_whisker_response.append(whisker_prob_response[ps_target_responders])
            pr_target_whisker_response.append(whisker_prob_response[pr_target_responders])

        #plot the result in boxplots for every animal across the two photostim types
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5), sharey=True);

        ax[0].boxplot(ps_target_whisker_response);
        ax[0].set_title('photostim_similar');
        ax[1].boxplot(pr_target_whisker_response);
        ax[1].set_title('photostim_random');