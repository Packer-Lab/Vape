## Analysis & visualisation code for decoding analysis
## Thijs van der Plas

import os, sys, pickle
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import xarray as xr
import scipy

## From Vape:
# import utils.ia_funcs as ia 
# import utils.utils_funcs as uf

## Data paths, from 'rob_setup_notebook.ipynb'
vape_path = '/home/tplas/repos/Vape/'
s2p_path = 'home/rlees/Documents/Code/suite2p'
qnap_data_path = '/home/rlees/mnt/qnap/Data' # for Ubuntu
qnap_path = qnap_data_path[:-5]

pkl_folder = os.path.join(qnap_path, 'pkl_files')
master_path = os.path.join(qnap_path, 'master_pkl', 'master_obj.pkl')
fig_save_path = os.path.join(qnap_path, 'Analysis', 'Figures')
stam_save_path = os.path.join(qnap_path, 'Analysis', 'STA_movies')
s2_borders_path = os.path.join(qnap_path, 'Analysis', 'S2_borders')

sess_type_dict = {'sens': 'sensory_2sec_test',
                  'proj': 'projection_2sec_test'}

sys.path.append(vape_path)
sys.path.append(os.path.join(vape_path, 'my_suite2p')) # to import ops from settings.py in that folder
sys.path.append(s2p_path)
import suite2p

def get_session_names(pkl_folder=pkl_folder,
                      sess_type='sens'):
    pkl_folder_path = os.path.join(pkl_folder, sess_type_dict[sess_type])
    list_session_names = os.listdir(pkl_folder_path)
    exclude_list = ['2020-09-09_RL100.pkl', '2020-09-15_RL102.pkl']
    list_session_names = [x for x in list_session_names if x not in exclude_list]
    assert len(list_session_names) == 6, 'less or more than 6 sessions found'
    return list_session_names

def load_session(pkl_folder=pkl_folder, 
                 sess_type='sens',
                 session_name=None,
                 session_id=0, verbose=1):
    if session_name is None and type(session_id) == int:
        list_session_names = get_session_names(pkl_folder=pkl_folder,
                    sess_type=sess_type)
        assert session_id < len(list_session_names)
        session_name = list_session_names[session_id]
        if verbose > 0:
            print('session name :', session_name)

    pkl_path = os.path.join(pkl_folder, sess_type_dict[sess_type], session_name)
    with open(pkl_path, 'rb') as f:
        ses_obj = pickle.load(f)

    if verbose > 1:
        # Show all attributes in ses_obj or exp_objs
        print('Session object attributes')
        for key, value in vars(ses_obj).items():
            print(key, type(value))

        print('\nExperimental object attributes')
        for key, value in vars(ses_obj.spont).items():
            print(key, type(value))

    return ses_obj, session_name 

class SimpleSession():
    """Class that stores session object in format that's easier to access for decoding analysis"""
    def __init__(self, sess_type='sens', session_id=0, verbose=1):
        self.sess_type = sess_type
        self.session_id = session_id
        self.verbose = verbose 

        self.SesObj, self.session_name = load_session(sess_type=self.sess_type,
                                session_id=self.session_id, verbose=self.verbose)
        
        if sess_type == 'sens':
            self._list_tt_original = ['photostim_s', 'photostim_r', 'spont', 'whisker_stim']
        elif sess_type == 'proj':
            self._list_tt_original = ['photostim_s', 'photostim_r', 'spont']

        self.sorted_inds_neurons = None  # default 

        self.filter_neurons()
        self.create_data_objects()
        self.create_full_dataset()
        # self.sort_neurons(sorting_method='normal')


    def filter_neurons(self):
        """Filter neurons that should not be used in any analysis. These are:

        - Neurons with max(abs(DF/F)) > 10 for any trial type
        """
        ## Find which neurons to keep:
        for i_tt, tt in enumerate(self._list_tt_original):
            data_obj = getattr(self.SesObj, tt)
            if i_tt == 0:  # make mat that stores max values 
                n_neurons = data_obj.all_trials[0].shape[0]
                mv_neurons_mat = np.zeros((n_neurons, len(self._list_tt_original)))
            else:
                assert n_neurons == data_obj.all_trials[0].shape[0]

            mv_neurons_mat[:, i_tt] = np.max(np.abs(data_obj.all_trials[0]), (1, 2)) # get max values

        if self.verbose > 1:
            print(mv_neurons_mat)
        mv_neurons_mat = np.max(mv_neurons_mat, 1)  # max across tt
        filter_neurons_arr = mv_neurons_mat > 10 
        mask_neurons = np.logical_not(filter_neurons_arr)  # flip (True = keep)
        if self.verbose > 0:
            print(f'Excluded {np.sum(filter_neurons_arr)} out of {len(filter_neurons_arr)} neurons')

        self._n_neurons_original = len(filter_neurons_arr)
        self._n_neurons_removed = np.sum(filter_neurons_arr)
        self._mask_neurons_keep = mask_neurons  
        # assert type(self._mask_neurons_keep[0]) == np.bool, type(self._mask_neurons_keep)
        self.n_neurons = np.sum(mask_neurons)
        

    def create_data_objects(self):
        """Put original data objects of self.SesObj in more convenient format that incorporates neuron filtering"""
        self.all_trials, self.targeted_cells = {}, {}
        self.n_targeted_cells, self._n_targeted_cells_original = {}, {} 
        self.cell_s1, self.cell_s2, self.cell_id= {}, {}, {}
        self.stim_dur, self.target_coords = {}, {}
        if self.sess_type == 'sens':
            self._key_dict = {'photostim_s': 'sensory', 
                              'photostim_r': 'random',
                              'whisker_stim': 'whisker',
                              'spont': 'sham'}
        elif sess_type == 'proj':
            self._key_dict = {'photostim_s': 'projecting', 
                              'photostim_r': 'non_projecting',
                              'spont': 'sham'}
        self.list_tt = [self._key_dict[tt] for tt in self._list_tt_original]
        for i_tt, tt in enumerate(self._list_tt_original):
            # print(tt)
            assert len(getattr(self.SesObj, tt).all_trials) == 1, 'more than 1 imaging plane detected ??'
            self.all_trials[self._key_dict[tt]] = getattr(self.SesObj, tt).all_trials[0][self._mask_neurons_keep, :, :]
            ## You could do the same for:
                ## all_amplitudes
                ## stats
                ## prob
                
            if i_tt == 0:
                self.time_array = getattr(self.SesObj, tt).time
                self.n_timepoints = len(self.time_array)
                assert self.n_timepoints == self.all_trials[self._key_dict[tt]].shape[1]
            else:
                assert np.isclose(self.time_array, getattr(self.SesObj, tt).time, atol=1e-6).all(), 'time arrays not equal across tts?'
            
            if tt != 'whisker_stim':  # targets not defined for whisker stim tt
                self.targeted_cells[self._key_dict[tt]] = getattr(self.SesObj, tt).targeted_cells[self._mask_neurons_keep]
                self.n_targeted_cells[self._key_dict[tt]] = np.sum(self.targeted_cells[self._key_dict[tt]])
                self._n_targeted_cells_original[self._key_dict[tt]] = np.sum(getattr(self.SesObj, tt).targeted_cells)
                self.target_coords[self._key_dict[tt]] = [x for i_x, x in enumerate(getattr(self.SesObj, tt).target_coords) if self._mask_neurons_keep[i_x]]
            
            self.cell_s1[self._key_dict[tt]] = np.array(getattr(self.SesObj, tt).cell_s1[0])[self._mask_neurons_keep]
            self.cell_s2[self._key_dict[tt]] = np.array(getattr(self.SesObj, tt).cell_s2[0])[self._mask_neurons_keep]
            self.cell_id[self._key_dict[tt]] = np.array(getattr(self.SesObj, tt).cell_id[0])[self._mask_neurons_keep]

            self.stim_dur[self._key_dict[tt]] = getattr(self.SesObj, tt).stim_dur

    def create_full_dataset(self):
        full_data = np.concatenate([self.all_trials[tt] for tt in self.list_tt], axis=2)  # concat across trials 
        assert full_data.shape[0] == self.n_neurons and full_data.shape[1] == self.n_timepoints 
        tt_arr = []
        for tt in self.list_tt:
            tt_arr += [tt] * self.all_trials[tt].shape[2]
        tt_arr = np.array(tt_arr)

        data_arr = xr.DataArray(full_data, dims=('neuron', 'time', 'trial'),
                                coords={'neuron': np.arange(full_data.shape[0]),  #could also do cell id but this is easier i think
                                        'time': self.time_array,
                                        'trial': np.arange(full_data.shape[2])})
        data_arr.time.attrs['units'] = 's'
        data_arr.neuron.attrs['units'] = '#'
        data_arr.trial.attrs['units'] = '#'
        data_arr.attrs['units'] = 'DF/F'

        data_set = xr.Dataset({'activity': data_arr, 
                               'cell_s1': ('neuron', self.cell_s1['sham']),  # same for all tt anyway
                               'cell_id': ('neuron', self.cell_id['sham']),
                               'trial_type': ('trial', tt_arr)})

        self.full_ds = data_set

       
    def dataset_selector(self, region=None, min_t=None, max_t=None, trial_type_list=None,
                         remove_added_dimensions=True, sort_neurons=True, reset_sort=False):
        """## xarray indexing cheat sheet:
        self.full_ds.activity.data  # retrieve numpy array type 
        self.full_ds.activity[0, :, :]  # use np indexing 

        self.full_ds.activity.isel(time=6)  # label-based index-indexing (ie the 6th time point)
        self.full_ds.activity.sel(trial=[6, 7, 8], neuron=55)  # label-based value-indexing (NB: can ONLY use DIMENSIONS, not other variables in dataset)
        self.full_ds.activity.sel(time=5, method='nearest')  # label-based value-indexing, finding the nearest match (good with floating errors etc)
        self.full_ds.sel(time=5, method='nearest')  # isel and sel can also be used on entire ds
        
        self.full_ds.activity.where(tmp.full_ds.bool_s1, drop=True)  # index by other data array; use drop to get rid of excluded data poitns 
        self.full_ds.where(tmp.full_ds.bool_s1, drop=True)  # or on entire ds (note that it works because bool_s1 is specified to be along 'neuron' dimension)
        self.full_ds.where(tmp.full_ds.time > 3, drop=True)  # works with any bool array
        self.full_ds.where(tmp.full_ds.neuron == 50, drop=True)  # dito 
        self.full_ds.where(tmp_full_ds.trial_type.isin(['sensory', 'random']), drop=True)  # use da.isin for multipel value checking
        """

        tmp_data = self.full_ds.copy(deep=True)
        if region is not None:
            assert region in ['s1', 's2']
            if region == 's1':
                tmp_data = tmp_data.where(tmp_data.cell_s1, drop=True)
            elif region == 's2':
                tmp_data = tmp_data.where(np.logical_not(tmp_data.cell_s1), drop=True)
                
        if min_t is not None:
            tmp_data = tmp_data.where(tmp_data.time >= min_t, drop=True)
        if max_t is not None:
            tmp_data = tmp_data.where(tmp_data.time <= max_t, drop=True)

        if trial_type_list is not None:
            assert type(trial_type_list) == list
            assert np.array([tt in self.list_tt for tt in trial_type_list]).all()
            tmp_data = tmp_data.where(tmp_data.trial_type.isin(trial_type_list), drop=True)

        ## squeeze trial type &  potentially other vars that got extra dims (cell_s1 cell_id)
        if remove_added_dimensions:
            ## Trial type:
            if 'neuron' in tmp_data.trial_type.dims:
                tmp_data = tmp_data.assign(trial_type=tmp_data.trial_type.isel(neuron=0).drop('neuron'))  # is this the best way? probably missing some magic here
            if 'time' in tmp_data.trial_type.dims:
                tmp_data = tmp_data.assign(trial_type=tmp_data.trial_type.isel(time=0).drop('time'))  
            ## S1 
            if 'time' in tmp_data.cell_s1.dims:
                tmp_data = tmp_data.assign(cell_s1=tmp_data.cell_s1.isel(time=0).drop('time')) 
            if 'trial' in tmp_data.cell_s1.dims:
                tmp_data = tmp_data.assign(cell_s1=tmp_data.cell_s1.isel(trial=0).drop('trial')) 
            ## Cell ID
            if 'time' in tmp_data.cell_id.dims:
                tmp_data = tmp_data.assign(cell_id=tmp_data.cell_id.isel(time=0).drop('time')) 
            if 'trial' in tmp_data.cell_id.dims:
                tmp_data = tmp_data.assign(cell_id=tmp_data.cell_id.isel(trial=0).drop('trial')) 


        # apply sorting (to neurons, and randomizatin of trials ? )
        if sort_neurons:
            if self.sorted_inds_neurons is None or reset_sort:
                if self.verbose > 0:
                    print('sorting neurons')
                _ = self.sort_neurons(data=tmp_data.activity.data.mean(2))
            tmp_data = tmp_data.assign(sorting_neuron_indices=('neuron', self.sorted_inds_neurons))  # add sorted indices on neuron dim
            tmp_data = tmp_data.sortby(tmp_data.sorting_neuron_indices)

        return tmp_data

    def sort_neurons(self, data=None, sorting_method='sum'):
        """from pop off"""
        if data is None:
            print('WARNING; using pre-specified data for sorting')
            data = self.all_trials['sensory'].mean(2)  # trial averaged
        else:
            assert data.ndim == 2
            # should be neurons x times 
            # TODO; do properly with self.full_data 
        if sorting_method == 'correlation':
            sorting = opt_leaf(data, link_metric='correlation')[0]
        elif sorting_method == 'euclidean':
            sorting = opt_leaf(data, link_metric='euclidean')[0]
        elif sorting_method == 'max_pos':
            arg_max_pos = np.argmax(data, 1)
            assert len(arg_max_pos) == data.shape[0]
            sorting = np.argsort(arg_max_pos)
        elif sorting_method == 'abs_max_pos':
            arg_max_pos = np.argmax(np.abs(data), 1)
            assert len(arg_max_pos) == data.shape[0]
            sorting = np.argsort(arg_max_pos)
        elif sorting_method == 'normal':
            return np.arange(data.shape[0])
        elif sorting_method == 'amplitude':
            max_val_arr = np.max(data, 1)
            sorting = np.argsort(max_val_arr)[::-1]
        elif sorting_method == 'sum':
            sum_data = np.sum(data, 1)
            sorting = np.argsort(sum_data)[::-1]
        if self.verbose > 0:
            print(f'Neurons sorted by {sorting_method}')
        self.sorted_inds_neurons = sorting
        return sorting



def opt_leaf(w_mat, dim=0, link_metric='correlation'):
    '''(from popoff)
    create optimal leaf order over dim, of matrix w_mat.
    see also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.optimal_leaf_ordering.html#scipy.cluster.hierarchy.optimal_leaf_ordering'''
    assert w_mat.ndim == 2
    if dim == 1:  # transpose to get right dim in shape
        w_mat = w_mat.T
    dist = scipy.spatial.distance.pdist(w_mat, metric=link_metric)  # distanc ematrix
    link_mat = scipy.cluster.hierarchy.ward(dist)  # linkage matrix
    if link_metric == 'euclidean':
        opt_leaves = scipy.cluster.hierarchy.leaves_list(scipy.cluster.hierarchy.optimal_leaf_ordering(link_mat, dist))
        # print('OPTIMAL LEAF SOSRTING AND EUCLIDEAN USED')
    elif link_metric == 'correlation':
        opt_leaves = scipy.cluster.hierarchy.leaves_list(link_mat)
    return opt_leaves, (link_mat, dist)


# def plot_single_raster_plot(data_mat, session, ax=None, cax=None, reg='S1', tt='hit', c_lim=0.2,
#                             imshow_interpolation='nearest', plot_cbar=False, print_ylabel=False,
#                             sort_tt_list='NA', n_trials=None, time_ticks=[], time_tick_labels=[],
#                             s1_lim=None, s2_lim=None, plot_targets=True, spec_target_trial=None,
#                             ol_neurons_s1=None, ol_neurons_s2=None, plot_yticks=True, transparent_art=False,
#                             plot_xlabel=True, n_stim=None, time_axis=None, filter_150_artefact=True,
#                             cbar_pad=1.02, target_tt_specific=True):

#     if ax is None:
#         ax = plt.subplot(111)

#     ## Plot artefact
#     if tt in ['hit', 'miss']:
#         if time_axis is None:
#             print('no time axis given to raster')
#             zero_tick = 120
#             ax.axvspan(zero_tick-2, zero_tick+30*0.5, alpha=1, color=color_tt['photostim'])
#         else:
#             # time_axis[np.logical_and(time_axis >= -0.07, time_axis < 0.35)] = np.nan
#             start_art_frame = np.argmin(np.abs(time_axis + 0.07))
#             if filter_150_artefact:
#                 end_art_frame = np.argmin(np.abs(time_axis - 0.35))
#             else:
#                 end_art_frame = np.argmin(np.abs(time_axis - 0.83))
#             if not transparent_art:
#                 data_mat = copy.deepcopy(data_mat)
#                 data_mat[:, start_art_frame:end_art_frame] = np.nan
#             ax.axvspan(start_art_frame - 0.25, end_art_frame - 0.25, alpha=0.3, color=color_tt['photostim'])

#     ## Plot raster plots
#     im = ax.imshow(data_mat, aspect='auto', vmin=-c_lim, vmax=c_lim,
#                     cmap='BrBG_r', interpolation=imshow_interpolation)

#     if plot_cbar:
#         if cax is None:
#             print('cax is none')
#             cbar = plt.colorbar(im, ax=ax).set_label(r"$\Delta F/F$" + ' activity')# \nnormalised per neuron')
#         else:
#             ## pretty sure shrink & cbar_pad dont work because cax is already defined.
#             cbar = plt.colorbar(im, cax=cax, orientation='vertical', shrink=0.5, pad=cbar_pad)
#             cbar.set_label(r"$\Delta F/F$", labelpad=3)
#             cbar.set_ticks([])
#             cbar.ax.text(0.5, -0.01, '-0.2'.replace("-", u"\u2212"), transform=cbar.ax.transAxes, va='top', ha='center')
#             cbar.ax.text(0.5, 1.0, '+0.2', transform=cbar.ax.transAxes, va='bottom', ha='center')       
    
#     if print_ylabel:
#         ax.set_ylabel(f'Neuron ID sorted by {reg}-{sort_tt_list}\npost-stim trial correlation',
#                       fontdict={'weight': 'bold'}, loc=('bottom' if n_stim is not None else 'center'))
#     if n_stim is None:
#         ax.set_title(f'Trial averaged {tt} {reg} (N={n_trials})')
#     else:
#         ax.set_title(f'{tt} {reg}, n_stim={n_stim} (N={n_trials})')
#     if plot_xlabel:
#         ax.set_xlabel(f'Time (s)')
#     ax.set_xticks(time_ticks)

#     ax.set_xticklabels(time_tick_labels)
#     if plot_yticks:
#         ax.tick_params(axis='y', left='on', which='major')
#         ax.yaxis.set_minor_locator(MultipleLocator(2))
#     else:
#         ax.set_yticks([])
#     # ax.tick_params(axis='y', left='on', which='minor', width=0.5)
#     if s1_lim is not None and reg == 'S1':
#         ax.set_ylim(s1_lim)
#     if s2_lim is not None and reg == 'S2':
#         ax.set_ylim(s2_lim)

#     ## Target indicator
#     if plot_targets and tt in ['hit', 'miss']:
#         if reg == 'S1':
#             reg_bool = session.s1_bool
#         elif reg == 'S2':
#             reg_bool = session.s2_bool
#         assert len(np.unique(session.is_target.mean((0, 1)))) == 1  # same for all time points
#         if filter_150_artefact:  # 150 not included
#             target_mat = session.is_target[:, session.photostim < 2, :]
#         else:
#             target_mat = session.is_target
#         if spec_target_trial is None: 
#             if target_tt_specific:  # get hit/miss specific targets
#                 if filter_150_artefact:
#                     tt_spec_arr = session.outcome[session.photostim < 2] == tt
#                 else:
#                     tt_spec_arr = session.outcome == tt
#                 target_mat = target_mat[:, tt_spec_arr, :]
#             neuron_targ = np.mean(target_mat, (1, 2))
#         else:
#             neuron_targ = np.mean(target_mat, 2)
#             neuron_targ = neuron_targ[:, spec_target_trial]
#         neuron_targ_reg = neuron_targ[reg_bool]  # select region
#         if reg == 'S1':
#             neuron_targ_reg = neuron_targ_reg[ol_neurons_s1]  # sort
#         elif reg == 'S2':
#             neuron_targ_reg = neuron_targ_reg[ol_neurons_s2]
#         divider = make_axes_locatable(ax)
#         targ_ax = divider.append_axes('right', size='6%', pad=0.0)
#         targ_ax.imshow(neuron_targ_reg[:, None], cmap='Greys', aspect='auto', interpolation='nearest')
#         targ_ax.set_xticks([])
#         targ_ax.set_yticks([])
#         if s1_lim is not None and reg == 'S1':
#             targ_ax.set_ylim(s1_lim)
#         if s2_lim is not None and reg == 'S2':
#             targ_ax.set_ylim(s2_lim)
#     return ax