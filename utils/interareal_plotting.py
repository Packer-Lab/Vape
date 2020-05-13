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
        
        # General attributes
        self.pkl_folder = pkl_folder
        self.pkl_name = []
        self.pkl_path = []
        
        # Experimental attributes
        self.sheet_name = []
        self.stim_type = []
        self.tiff_path = []
        self.fps = []
        self.n_units = []
        self.cell_id = []
        self.cell_med = []
        self.s1_cell = []
        self.s2_cell = []
        self.num_s1_cells = []
        self.num_s2_cells = []
        self.stim_dur = []
        self.stim_freq = [] 
        self.pre_frames = []
        self.post_frames = []
        self.duration_frames = []
        self.time = []
        self.all_trials = []
        self.stas = []
        self.trial_sig_dff = []
        self.trial_sig_dfsf = []
        self.sta_sig = []
        self.sta_sig_nomulti = []
        self.prob_response = []
        self.sta_amplitudes = []
        self.all_amplitudes = []
        
        # Photostim attributes
        self.targeted_cells = []
        self.target_coords = []
        self.n_targets = []
        self.n_targeted_cells = []
        self.trial_target_dff = []
        self.trial_w_targets = []
        self.trial_euclid_dist = []
        self.sta_euclid_dist = []
        
        # Populate the lists above
        self.addPickles()

        
    def _parseGeneralMetadata(self, exp_obj):
        
        # NOTE: only taking the first plane [0]
        
        # Variables and attributes that are shared within a session
        self.n_units.append(exp_obj.n_units[0])
        self.cell_id.append(exp_obj.cell_id[0])
        self.num_s1_cells.append(exp_obj.num_s1_cells[0])
        self.num_s2_cells.append(exp_obj.num_s2_cells[0])
        self.cell_med.append(np.array(exp_obj.cell_med[0]))
        self.s1_cell.append(np.array(exp_obj.cell_s1[0]))
        self.s2_cell.append(np.array(exp_obj.cell_s2[0]))
        
        # Variables and attributes that differ within a session
        # Should either be made in to an additional list (attributes) 
        # or be expanded to fit the length of that dimension of the data (variables)
        
        # Attributes (things I won't use to index)
        self.tiff_path.append(os.path.split(exp_obj.tiff_path)[1])
        self.fps.append(exp_obj.fps)
        self.pre_frames.append(exp_obj.pre_frames)
        self.post_frames.append(exp_obj.post_frames)
        self.duration_frames.append(exp_obj.duration_frames)
        self.stim_dur.append(exp_obj.stim_dur)
        self.stim_freq.append(exp_obj.stim_freq)
        
        # Variables
        self.all_trials.append(exp_obj.all_trials[0]) # [cell x frame x trial]
        self.time.append(exp_obj.time) # [frame]
        self.stim_type.append(exp_obj.stim_type) # expand this to repeat for len(trials)        
        self.stas.append(np.array(exp_obj.stas[0])) # [cell x frame]
        self.trial_sig_dff.append(np.array(exp_obj.trial_sig_dff[0])) # [cell x trial]
        self.trial_sig_dfsf.append(np.array(exp_obj.trial_sig_dfsf[0])) # [cell x trial]
        self.sta_sig.append(np.array(exp_obj.sta_sig[0])) # [cell]
        self.sta_sig_nomulti.append(np.array(exp_obj.sta_sig_nomulti[0])) # [cell]
        self.prob_response.append(np.array(exp_obj.prob_response[0])) # [cell]
        self.sta_amplitudes.append(np.array(exp_obj.sta_amplitudes[0])) # [cell]
        self.all_amplitudes.append(np.array(exp_obj.all_amplitudes[0])) # [cell x trial]

    
    def _parsePhotostimMetadata(self, exp_obj):
        
        if any(s in exp_obj.stim_type for s in ['pr', 'ps', 'none']):
            self.targeted_cells.append(exp_obj.targeted_cells)
            self.target_coords.append(exp_obj.target_coords)
            self.n_targets.append(exp_obj.n_targets)
            self.n_targeted_cells.append(exp_obj.n_targeted_cells)
            self.trial_target_dff.append(exp_obj.trial_target_dff)
            self.trial_w_targets.append(exp_obj.trial_w_targets)
            self.trial_euclid_dist.append(exp_obj.trial_euclid_dist)
            self.sta_euclid_dist.append(exp_obj.sta_euclid_dist)
        else:
            self.targeted_cells.append(False)
            self.target_coords.append(False)
            self.n_targets.append(False)
            self.n_targeted_cells.append(False)
            self.trial_target_dff.append(False)
            self.trial_w_targets.append(False)
            self.trial_euclid_dist.append(False)
            self.sta_euclid_dist.append(False)

    
    def _performAnalysis(self):

        for pkl_file in self.new_pkls:
            print('Collecting analysed data for pickled object:', pkl_file, '          ', end='\r')
            whisker_cells = False
            
            basename = os.path.basename(pkl_file)
            self.pkl_name.append(basename)

            with open(pkl_file, 'rb') as f:
                ses_obj = pickle.load(f)
            
            exp_list = [ses_obj.photostim_r, ses_obj.photostim_s]

            if ses_obj.spont.n_frames > 0:
                exp_list.append(ses_obj.spont)
                
            if ses_obj.whisker_stim.n_frames > 0:
                exp_list.append(ses_obj.whisker_stim)
                whisker_cells = np.where(ses_obj.whisker_stim.sta_sig[0]) # find the number of whisker responsive cells targeted on each trial

            for exp_obj in exp_list:
                
                self.sheet_name.append(ses_obj.sheet_name)

                self._parseGeneralMetadata(exp_obj)
                
                self._parsePhotostimMetadata(exp_obj)

                
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
                ax[plot_index].set_xlim(np.amin(x), np.amax(x))
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
#                     z = np.polyfit(x, y, 1)
#                     p = np.poly1d(z)
#                     ax[plot_index].plot(x,p(x))
    
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
        
        exp_names = []
        
        #iterate through the groups
        for name, group in all_whisker_groups:

            #find the targets responding on each trial (boolean list)
            ps_target_responders = np.array(group.loc[group['stim_type'] == 'ps', 'target_responders'])[0]
            pr_target_responders = np.array(group.loc[group['stim_type'] == 'pr', 'target_responders'])[0]  

            #print the number of targets responding out of total number of targeted cells
            print(name)
            exp_names.append(name)
            print('Similar:', list(ps_target_responders).count(True), 'out of', int(group.loc[group['stim_type'] == 'ps', 'n_targeted_cells']))
            print('Random:', list(pr_target_responders).count(True), 'out of', int(group.loc[group['stim_type'] == 'pr', 'n_targeted_cells']))

            #find the probability of response for all cells to whisker stim
            whisker_prob_response = np.array(group.loc[group['stim_type'] == 'w', 'prob_response'])[0]

            #append the probability of response to whisker stim for the responding target cells (STA)  
            ps_target_whisker_response.append(whisker_prob_response[ps_target_responders])
            pr_target_whisker_response.append(whisker_prob_response[pr_target_responders])

        #plot the result in boxplots for every animal across the two photostim types
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5), sharey=True);
        
        for i, resps in enumerate([ps_target_whisker_response, pr_target_whisker_response]):
            dict_resp = {'exp_name' : exp_names, 'p_resp' : resps}
            df_resp = pd.DataFrame.from_dict(dict_resp)
        
            lst_col = 'p_resp'

            ps_df = pd.DataFrame({
                    col:np.repeat(df_resp[col].values, df_resp[lst_col].str.len())
                    for col in df_resp.columns.drop(lst_col)}
                ).assign(**{lst_col:np.concatenate(df_resp[lst_col].values)})[df_resp.columns]
        
            plt.sca(ax[i])
            sns.boxplot(x='exp_name', y='p_resp', data=ps_df, width=0.2)
            sns.swarmplot(x='exp_name', y='p_resp', data=ps_df, color='k', size=5)
            plt.xticks(
                rotation=45, 
                horizontalalignment='right',
                fontweight='light',
                fontsize='x-large'  
            )