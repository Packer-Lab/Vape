## Analysis & visualisation code for decoding analysis
## Thijs van der Plas

from json import decoder
import os, sys, pickle

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

from select import select
from urllib import response
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
# from utils.utils_funcs import correct_s2p_combined 
import xarray as xr
import scipy
from profilehooks import profile, timecall
import sklearn.discriminant_analysis, sklearn.model_selection, sklearn.decomposition

## From Vape:
# import utils.ia_funcs as ia 
# import utils.utils_funcs as uf

sess_type_dict = {'sens': 'sensory_2sec_test',
                  'proj': 'projection_2sec_test'}

sys.path.append(vape_path)
sys.path.append(os.path.join(vape_path, 'my_suite2p')) # to import ops from settings.py in that folder
sys.path.append(s2p_path)
import suite2p

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
colour_tt_dict = {'sensory': colors[0],
                  'random': colors[1],
                  'projecting': colors[2],
                  'non_projecting': colors[3],
                  'whisker': colors[4],
                  'sham': 'k'}
label_tt_dict = {'sensory': 'Sensory',
                  'random': 'Random',
                  'projecting': 'Projecting',
                  'non_projecting': 'Non-projecting',
                  'whisker': 'Whisker',
                  'sham': 'Sham'}                 

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
    def __init__(self, sess_type='sens', session_id=0, verbose=1,
                 shuffle_trial_labels=False, shuffle_timepoints=False, 
                 shuffle_all_data=False,
                 bool_filter_neurons=True):
        self.sess_type = sess_type
        self.session_id = session_id
        self.verbose = verbose 
        self.shuffle_timepoints = shuffle_timepoints
        self.shuffle_trial_labels = shuffle_trial_labels
        self.shuffle_all_data = shuffle_all_data
        self.bool_filter_neurons = bool_filter_neurons

        self.SesObj, self.session_name = load_session(sess_type=self.sess_type,
                                session_id=self.session_id, verbose=self.verbose)
        self.session_name_readable = self.session_name.rstrip('.pkl')
        
        if sess_type == 'sens':
            self._list_tt_original = ['photostim_s', 'photostim_r', 'spont', 'whisker_stim']
        elif sess_type == 'proj':
            self._list_tt_original = ['photostim_s', 'photostim_r', 'spont']

        self.sorted_inds_neurons = None  # default 

        self.filter_neurons(filter_max_abs_dff=self.bool_filter_neurons)
        self.create_data_objects()
        self.create_full_dataset()
        # self.sort_neurons(sorting_method='normal')

    def filter_neurons(self, filter_max_abs_dff=True):
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
        if filter_max_abs_dff:
            filter_neurons_arr = mv_neurons_mat > 10 
        else:
            filter_neurons_arr = np.zeros(n_neurons, dtype=bool)
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

        ## concatenating all trials
        """trials are done in block format; 100 trials per type. 
        Do in 4 blocks (no interleaving). Whisker, sham, PS1 PS2. (swap last two every other experiment). 
        ITI 15 seconds, (but data is defined as 12 second trials)"""
        if self.sess_type == 'sens':
            self._key_dict = {'photostim_s': 'sensory', 
                              'photostim_r': 'random',
                              'whisker_stim': 'whisker',
                              'spont': 'sham'}
        elif self.sess_type == 'proj':
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

    def create_full_dataset(self, zscore=False):
        full_data = np.concatenate([self.all_trials[tt] for tt in self.list_tt], axis=2)  # concat across trials 
        assert full_data.shape[0] == self.n_neurons and full_data.shape[1] == self.n_timepoints 
        tt_arr = []
        for tt in self.list_tt:
            tt_arr += [tt] * self.all_trials[tt].shape[2]
        tt_arr = np.array(tt_arr)
        assert len(tt_arr) == full_data.shape[2]

        if zscore:
            full_data = (full_data - full_data.mean((0, 1))) / full_data.std((0, 1))
            print('WARNING: z-scoring messes up the baselining (that average pre-stim activity =0)')
        else:
            ## assert baselining is as expected
            baseline_frames = self.SesObj.spont.pre_frames  # same for any trial type
            assert np.abs(full_data[:, :baseline_frames, :].mean()) < 1e-8  # mean across neurons and pre-stim time points and trials
            assert np.max(np.abs(full_data[:, :baseline_frames, :].mean((0, 1)))) < 1e-8  # mean across neurons and pre-stim time points

        if self.shuffle_trial_labels:
            # random_inds = np.random.permutation(full_data.shape[2])
            trial_inds = np.arange(full_data.shape[2])
            # full_data = full_data[:, :, random_inds]
            # for ineuron in range(full_data.shape[0]):
            for itp in range(full_data.shape[1]):
                random_trial_inds = np.random.permutation(full_data.shape[2])
                full_data[:, itp, :] = full_data[:, itp, :][:, random_trial_inds]
            if self.verbose > 0:
                print('WARNING: trials labels are shuffled!')
        else:
            trial_inds = np.arange(full_data.shape[2])

        if self.shuffle_timepoints:
            n_trials = full_data.shape[2]
            for it in range(n_trials):
                random_tp_inds = np.random.permutation(full_data.shape[1])
                full_data[:, :, it] = full_data[:, random_tp_inds, it]
            # for it in range(n_trials):
            #     random_neuron_inds = np.random.permutation(full_data.shape[0])
            #     full_data[:, :, it] = full_data[random_neuron_inds, :, it]
                # random_tp_inds = np.random.permutation(full_data.shape[1])
                # full_data[:, :, it] = full_data[:, random_tp_inds, it]
            if self.verbose > 0:
                print('WARNING: time points are shuffled per trial')

        if self.shuffle_all_data:
            full_data_shape = full_data.shape 
            full_data = full_data.ravel()
            np.random.shuffle(full_data)
            full_data = full_data.reshape(full_data_shape)
            # full_data = np.random.randn(full_data.shape[0], full_data.shape[1], full_data.shape[2])  # totally white noise 

            print('WARNING: all data points shuffled!')

        data_arr = xr.DataArray(full_data, dims=('neuron', 'time', 'trial'),
                                coords={'neuron': np.arange(full_data.shape[0]),  #could also do cell id but this is easier i think
                                        'time': self.time_array,
                                        'trial': trial_inds})
        data_arr.time.attrs['units'] = 's'
        data_arr.neuron.attrs['units'] = '#'
        data_arr.trial.attrs['units'] = '#'
        data_arr.attrs['units'] = 'DF/F'

        data_set = xr.Dataset({'activity': data_arr, 
                               'cell_s1': ('neuron', self.cell_s1['sham']),  # same for all tt anyway
                               'cell_id': ('neuron', self.cell_id['sham']),
                               'trial_type': ('trial', tt_arr)})

        self.full_ds = data_set
        self.time_aggr_ds = None

    def dataset_selector(self, region=None, min_t=None, max_t=None, trial_type_list=None,
                         remove_added_dimensions=True, sort_neurons=True, reset_sort=False,
                         deepcopy=True):
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
        if deepcopy:
            tmp_data = self.full_ds.copy(deep=True)
        else:
            tmp_data = self.full_ds
        if region is not None:
            assert region in ['s1', 's2']
            if region == 's1':
                tmp_data = tmp_data.where(tmp_data.cell_s1, drop=True)
            elif region == 's2':
                tmp_data = tmp_data.where(np.logical_not(tmp_data.cell_s1), drop=True)

        ## TODO; speed up time selection   
        if min_t is not None:
            tmp_data = tmp_data.where(tmp_data.time >= min_t, drop=True)
        if max_t is not None:
            tmp_data = tmp_data.where(tmp_data.time <= max_t, drop=True)

        if trial_type_list is not None:
            assert type(trial_type_list) == list
            assert np.array([tt in self.list_tt for tt in trial_type_list]).all(), f'{trial_type_list} not in {self.list_tt}'
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
                assert tmp_data.activity.data.mean(2).shape[0] == len(tmp_data.neuron)
                sorting, _ = self.sort_neurons(data=tmp_data.activity.data.mean(2), sorting_method='correlation')
                assert len(sorting) == len(tmp_data.neuron)
                assert len(sorting) == len(self.sorted_inds_neurons)
                assert len(sorting) == len(self.sorted_inds_neurons_inverse)
            tmp_data = tmp_data.assign(sorting_neuron_indices=('neuron', self.sorted_inds_neurons_inverse))  # add sorted indices on neuron dim
            tmp_data = tmp_data.sortby(tmp_data.sorting_neuron_indices)

        return tmp_data

    def sort_neurons(self, data=None, sorting_method='sum', save_sorting=True):
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
        tmp_rev_sort = np.zeros_like(sorting)
        for old_ind, new_ind in enumerate(sorting):
            tmp_rev_sort[new_ind] = old_ind
        if save_sorting:
            self.sorted_inds_neurons = sorting
            self.sorted_inds_neurons_inverse = tmp_rev_sort
        return sorting, tmp_rev_sort

    def create_time_averaged_response(self, t_min=0.4, t_max=2, 
                        region=None, aggregation_method='average',
                        sort_neurons=False, subtract_pop_av=False,
                        subtract_pcs=False,
                        trial_type_list=None):
        """region: 's1', 's2', None [for both]"""
        selected_ds = self.dataset_selector(region=region, min_t=t_min, max_t=t_max,
                                    sort_neurons=False, remove_added_dimensions=True,
                                    trial_type_list=trial_type_list)  # all trial types
        
        if subtract_pcs:
            n_timepoints_per_trial = len(selected_ds.time)
            n_trials = len(selected_ds.trial)
            selected_ds_2d = xr.concat([selected_ds.sel(trial=i_trial) for i_trial in selected_ds.trial], dim='time')  # concat trials on time axis
            # lfa = sklearn.decomposition.FactorAnalysis(n_components=2)
            lfa = sklearn.decomposition.PCA(n_components=3)
            activity_fit = selected_ds_2d.activity.data  # neurons x times
            activity_fit = activity_fit.transpose()
            pc_activity = lfa.fit_transform(X=activity_fit)
            print(lfa.explained_variance_ratio_)
            activity_neurons_proj_pcs = np.dot(pc_activity, lfa.components_)  # dot prod of PCA activity x loading
            activity_neurons_proj_pcs = activity_neurons_proj_pcs.transpose()
            activity_neurons_proj_pcs_3d = np.stack([activity_neurons_proj_pcs[:, (i_trial * n_timepoints_per_trial):((i_trial + 1) * n_timepoints_per_trial)] for i_trial in range(n_trials)], 
                                                    axis=2)
            # print(f'Subtracted LFA.')
            # selected_ds = selected_ds.assign(activity=selected_ds.activity - activity_neurons_proj_pcs_3d)
            print(f'Showing top 3 PCs')
            selected_ds = selected_ds.assign(activity=(('neuron', 'time', 'trial'), activity_neurons_proj_pcs_3d))
        if aggregation_method == 'average':
            tt_arr = selected_ds.trial_type.data  # extract becauses it's an arr of str, and those cannot be meaned (so will be dropped)
            selected_ds = selected_ds.mean('time')
            selected_ds = selected_ds.assign(trial_type=('trial', tt_arr))  # put back
        elif aggregation_method == 'max':
            tt_arr = selected_ds.trial_type.data  # extract becauses it's an arr of str, and those cannot be meaned (so will be dropped)
            selected_ds = selected_ds.max('time')
            selected_ds = selected_ds.assign(trial_type=('trial', tt_arr))  # put back
        else:
            print(f'WARNING: {aggregation_method} method not implemented!')
        
        if subtract_pop_av:
            selected_ds = selected_ds.assign(activity=selected_ds.activity - selected_ds.activity.mean('neuron'))

        if sort_neurons:
            self.sort_neurons(data=selected_ds.activity, 
                            sorting_method='correlation')
            selected_ds = selected_ds.assign(sorting_neuron_indices=('neuron', self.sorted_inds_neurons_inverse))  # add sorted indices on neuron dim
            selected_ds = selected_ds.sortby(selected_ds.sorting_neuron_indices)
        
        self.time_aggr_ds = selected_ds
        self.time_aggr_ds_pop_av_subtracted = subtract_pop_av
        return selected_ds
        
    def find_discr_index_neurons(self, tt_1='sensory', tt_2='sham'):

        assert self.time_aggr_ds is not None 
        mean_1 = self.time_aggr_ds.activity.where(self.time_aggr_ds.trial_type==tt_1, drop=True).mean('trial')
        mean_2 = self.time_aggr_ds.activity.where(self.time_aggr_ds.trial_type==tt_2, drop=True).mean('trial')
        
        var_1 = self.time_aggr_ds.activity.where(self.time_aggr_ds.trial_type==tt_1, drop=True).var('trial')
        var_2 = self.time_aggr_ds.activity.where(self.time_aggr_ds.trial_type==tt_2, drop=True).var('trial')

        dprime = np.abs(mean_1 - mean_2) / np.sqrt(var_1 + var_2)
        name = f'dprime_{tt_1}_{tt_2}'

        self.time_aggr_ds = self.time_aggr_ds.assign(**{name: ('neuron', dprime)})
        return dprime

    def find_discr_index_neurons_shuffled(self, tt_1='sensory', tt_2='sham'):

        assert self.time_aggr_ds is not None 
        responses_both = self.time_aggr_ds.activity.where(self.time_aggr_ds.trial_type.isin([tt_1, tt_2]), drop=True)
        responses_both = responses_both.copy(deep=True)
        responses_both = responses_both.data  # go to numpy format to make sure shuffling works 
        responses_both_shuffled = shuffle_along_axis(a=responses_both, axis=1)  # shuffle along trials, per neuron

        # print('normal: ')
        # print(responses_both.mean(1)[:20])
        # print('\n shuffled:')
        # print(responses_both_shuffled.mean(1)[:20])
        assert np.isclose(responses_both_shuffled.mean(1), responses_both.mean(1), atol=1e-5).all()  # ensure shuffling is done along trial dim only 
        assert responses_both_shuffled.shape[1] == 200, 'not 100 trials per tt??'
        responses_1 = responses_both_shuffled[:, :100]
        responses_2 = responses_both_shuffled[:, 100:]

        mean_1 = responses_1.mean(1)
        mean_2 = responses_2.mean(1)

        var_1 = responses_1.var(1)
        var_2 = responses_2.var(1)

        dprime = np.abs(mean_1 - mean_2) / np.sqrt(var_1 + var_2)
        name = f'dprime_{tt_1}_{tt_2}_shuffled'

        self.time_aggr_ds = self.time_aggr_ds.assign(**{name: ('neuron', dprime)})
        return dprime

    def find_all_discr_inds(self, region='s2', shuffled=False,
                            subtract_pop_av=False):
        if self.verbose > 0:
            print('Creating time-aggregate data set')
        ## make time-averaged data
        self.create_time_averaged_response(sort_neurons=False, region=region,
                                           subtracts_pop_av=subtract_pop_av)
        ## get discr arrays (stored in time_aggr_ds)
        if self.verbose > 0:
            print('Calculating d prime values')
        for tt in self.list_tt:  # get all comparisons vs sham
            if tt != 'sham':
                self.find_discr_index_neurons(tt_1=tt, tt_2='sham')
                if shuffled:
                    self.find_discr_index_neurons_shuffled(tt_1=tt, tt_2='sham')

    def population_tt_decoder(self, region='s2', bool_subselect_neurons=False,
                              decoder_type='LDA', tt_list=['whisker', 'sham'],
                              n_cv_splits=5, verbose=1, subtract_pcs=False):
        """Decode tt from pop of neurons.
        Use time av response, region specific, neuron subselection.
        Use CV, LDA?, return mean test accuracy"""
        ## make time-averaged data
        self.create_time_averaged_response(sort_neurons=False, region=region,
                                            subtract_pcs=subtract_pcs,
                                           subtract_pop_av=False, trial_type_list=tt_list)
        if verbose > 0:
            print('Time-aggregated activity object created')
        ## activity is now in self.time_aggr_ds

        ## neuron subselection
        assert bool_subselect_neurons is False, 'neuron sub selection not yet implemented'

        ## Prepare data
        tt_labels = self.time_aggr_ds.trial_type.data
        neural_data = self.time_aggr_ds.activity.data.transpose()
        assert tt_labels.shape[0] == neural_data.shape[0]  # number of trials 

        ## prepare CV
        cv_obj = sklearn.model_selection.StratifiedKFold(n_splits=n_cv_splits)
        score_arr = np.zeros(n_cv_splits)

        ## run decoder:
        i_cv = 0
        for train_index, test_index in cv_obj.split(X=neural_data, y=tt_labels):
            if verbose > 0:
                print(f'Decoder Cv loop {i_cv + 1}/{n_cv_splits}')
            neural_train, neural_test = neural_data[train_index, :], neural_data[test_index, :]
            tt_train, tt_test = tt_labels[train_index], tt_labels[test_index]

            ##  select decoder 
            if decoder_type == 'LDA':
                decoder_model = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
            elif decoder_type == 'QDA':
                decoder_model = sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis()
            elif decoder_type == 'logistic_regression':
                decoder_model = sklearn.linear_model.LogisticRegression()
            else:
                assert False, 'NOT IMPLEMENTED'

            decoder_model.fit(X=neural_train, y=tt_train)
            score_arr[i_cv] = decoder_model.score(X=neural_test, y=tt_test)
            if verbose > 0:
                print(f'Score: {score_arr[i_cv]}')
            i_cv += 1

        if verbose > 0:
            print(score_arr)

        return score_arr

        ##################{'whisker': array([0.94 , 0.615, 0.81 , 0.83 , 0.905, 0.665]), 'sensory': array([0.56 , 0.49 , 0.515, 0.62 , 0.505, 0.575]), 'random': array([0.5  , 0.41 , 0.5  , 0.5  , 0.49 , 0.485])}

def shuffle_along_axis(a, axis):
    ## https://stackoverflow.com/questions/5040797/shuffling-numpy-array-along-a-given-axis
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)

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

def despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def create_time_axis(ax, time_arr, axis='x', label_list=[-2, 0, 2, 4, 6, 8, 10], rotation=0):
    '''Works with heatmap, where len(time_arr) is the number of data points on that axis.
    Not sure how to capture this in an assert.. ?'''
    if axis == 'x':
        ax.set_xticks([np.argmin(np.abs(time_arr - x)) for x in label_list])
        ax.set_xticklabels(label_list, rotation=rotation);
    elif axis == 'y':
        ax.set_yticks([np.argmin(np.abs(time_arr - x)) for x in label_list])
        ax.set_yticklabels(label_list, rotation=rotation);
    else:
        print(f'WARNING: axis {axis} not recognised when creating time axis')


def plot_pop_av(Ses=None, ax_list=None, region_list=['s2'], sort_trials_per_tt=False,
                plot_trial_av=False):
    pop_act_dict = {}
    if ax_list is None:
        if plot_trial_av:
            fig, ax = plt.subplots(2 * len(region_list), len(Ses.list_tt), 
                                    figsize=(len(Ses.list_tt) * 5, 6 * len(region_list)),
                                    gridspec_kw={'wspace': 0.6, 'hspace': 0.5})

        else:   
            fig, ax = plt.subplots(len(region_list), len(Ses.list_tt), 
                                    figsize=(len(Ses.list_tt) * 5, 3 * len(region_list)),
                                    gridspec_kw={'wspace': 0.6, 'hspace': 0.5})
    for i_r, region in enumerate(region_list):
        if len(region_list) > 1:
            ax_row = ax[i_r]
        else:
            ax_row = ax
        for i_tt, tt in enumerate(Ses.list_tt):
            pop_act_dict[tt] = Ses.dataset_selector(region=region, trial_type_list=[tt], 
                                    remove_added_dimensions=True, sort_neurons=False)
            plot_data = pop_act_dict[tt].activity.mean('neuron').transpose()
            if sort_trials_per_tt:
                sorting, _ = Ses.sort_neurons(data=plot_data.data, save_sorting=False)  # hijack this function, don't save in class
                time_ax = plot_data.time.data
                plot_data = plot_data.data[sorting, :]  # trials sorted 
                sns.heatmap(plot_data, ax=ax_row[i_tt], vmin=-0.5, vmax=0.5, cmap='BrBG',
                            cbar_kws={'label': 'activity'})
                ax_row[i_tt].set_xlabel('time [s]')
                ax_row[i_tt].set_ylabel('sorted trials [#]')
                # ax_row[i_tt].inverse_yaxis()
                xtl = [-2, 0, 2, 4, 6, 8, 10]
                ax_row[i_tt].set_xticks([np.argmin(np.abs(time_ax - x)) for x in xtl])
                ax_row[i_tt].set_xticklabels(xtl, rotation=0)
            else:
                plot_data.plot(ax=ax_row[i_tt], vmin=-0.5)
            ax_row[i_tt].set_title(f'{tt} {region} population average')
            if plot_trial_av:
                if sort_trials_per_tt:
                    pass
                else:
                    plot_data = plot_data.data
                ax_av = ax[i_r + 2, i_tt]
                ax_av.plot(time_ax, plot_data.mean(0), c=colour_tt_dict[tt], linewidth=2)
                ax_av.set_xlabel('Time (s)')
                ax_av.set_ylabel('Trial-av, pop-av DF/F')
                ax_av.set_title(f'{tt} {region} trial-average')
                despine(ax_av)
    plt.suptitle(f'Session {Ses.session_name_readable}')

def plot_hist_discr(Ses=None, ax=None, max_dprime=None, plot_density=True,
                    plot_shuffled=True, yscale_log=False, show_all_shuffled=False,
                    plot_hist=True, plot_kde=False):
    if ax is None:
        ax = plt.subplot(111)

    if max_dprime is None:
        max_dprime = 0
        for tt in Ses.list_tt:
            if tt != 'sham':  # comparison with sham 
                arr_dprime = getattr(Ses.time_aggr_ds, f'dprime_{tt}_sham').data
                max_dprime = np.maximum(max_dprime, arr_dprime.max())
        max_dprime = np.maximum(max_dprime, 2)

    if (Ses.time_aggr_ds.cell_s1 == 1).all():
        region = 'S1'
    elif (Ses.time_aggr_ds.cell_s1 == 0).all():
        region = 'S2'
    else:
        region = 'S1 and S2'

    ## Assume discr has already been computed 
    bins = np.linspace(0, max_dprime * 1.01, 100)

    arr_dprime_dict = {}
    arr_dprime_shuffled_dict = {}
    kde_log_corr = 1e-6 if yscale_log else 0
    plot_tt_list = [tt for tt in Ses.list_tt if tt != 'sham']
    for i_tt, tt in enumerate(plot_tt_list):
        arr_dprime_dict[tt] = getattr(Ses.time_aggr_ds, f'dprime_{tt}_sham').data
        if plot_hist:
            _ = ax.hist(arr_dprime_dict[tt], bins=bins, alpha=0.7,
                    histtype='step', edgecolor=colour_tt_dict[tt],
                    linewidth=3, density=plot_density, label=f'{tt} vs sham')
        if plot_kde:
            kde_f = scipy.stats.gaussian_kde(arr_dprime_dict[tt])
            x_array = np.linspace(0, max_dprime * 1.01, 1000)
            ax.plot(x_array, kde_f(x_array) + kde_log_corr, c=colour_tt_dict[tt], linewidth=3,
                    alpha=0.7, label=f'{tt} vs sham')

        if plot_shuffled:
            if hasattr(Ses.time_aggr_ds, f'dprime_{tt}_sham_shuffled'):
                if show_all_shuffled or i_tt == 0:  # show only first one if not all
                    arr_dprime_shuffled_dict[tt] = getattr(Ses.time_aggr_ds, f'dprime_{tt}_sham_shuffled').data
                    if plot_hist:
                        _ = ax.hist(arr_dprime_shuffled_dict[tt], bins=bins, alpha=1,
                                histtype='step', edgecolor='grey', 
                                linewidth=3, density=plot_density, label=f'shuffled {tt} vs sham')
                    if plot_kde:
                        kde_f = scipy.stats.gaussian_kde(arr_dprime_shuffled_dict[tt])
                        x_array = np.linspace(0, max_dprime * 1.01, 1000)
                        ax.plot(x_array, kde_f(x_array) + kde_log_corr, c='grey', linewidth=3,
                                alpha=0.7, label=f'shuffled {tt} vs sham')
            else:
                print("Shuffled discr not found!")

    ax.legend(loc='best', frameon=False)
    ax.set_title(f'Discriminating trial types in {region} of\n{Ses.session_name_readable}')
    ax.set_xlabel('d prime')
    if plot_density:
        ax.set_ylabel('density')
    else:
        ax.set_ylabel('number of cells')
    if yscale_log:
        ax.set_yscale('log')
    despine(ax)

def plot_raster_sorted_activity(Ses=None, sort_here=False, create_new_time_aggr_data=False,
                                region='s2', ## region only applicable to creating new data!!
                                plot_trial_type_list=['whisker', 'sham'], verbose=1):
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    if create_new_time_aggr_data or (Ses.time_aggr_ds is None):
        ## make time-averaged data
        if verbose > 0:
            print('Creating time aggregated data')
        Ses.create_time_averaged_response(sort_neurons=False, region=region,
                                           subtract_pop_av=False)
    plot_data = Ses.time_aggr_ds.activity.where(Ses.time_aggr_ds.trial_type.isin(plot_trial_type_list), drop=True)
    print(plot_data)
    plot_data = plot_data.data

    tt_list = []
    for tt in Ses.time_aggr_ds.trial_type:
        if str(tt.data) not in tt_list:
            tt_list.append(str(tt.data))

    print(tt_list)

    if sort_here:
        sorted_inds, _ = Ses.sort_neurons(data=plot_data, sorting_method='euclidean')
        # print(sorted_inds)
        plot_data = plot_data[sorted_inds, :]
    sns.heatmap(plot_data, ax=ax, vmin=-0.5, vmax=0.5, 
                cmap='BrBG')
    ax.set_xlabel('Trial')
    ax.set_xticks(list(np.arange(2 * len(plot_trial_type_list) + 1) * 50))
    ax.set_xticklabels(sum([[x * 100, tt_list[x]] for x in range(len(plot_trial_type_list))], []) + [100 * (len(plot_trial_type_list) + 1)], rotation=0)
    if sort_here:
        ax.set_ylabel('Neurons (sorted by Eucl. distance')
    else:
        ax.set_ylabel('neuron')
    ax.set_title('Time-averaged data (average of 2 seconds post-stimulus per trial per neuron)')
    return ax

def bar_plot_decoder_accuracy(scores_dict, dict_sess_type_tt=None):

    if dict_sess_type_tt is None:
        dict_sess_type_tt = {'sens': ['sensory', 'random', 'whisker'],
                             'proj': ['projecting', 'non_projecting']}
    n_tt = 5
    fig, ax = plt.subplots(1, 2, figsize=(8, 3), gridspec_kw={'wspace': 0.5})

    for i_r, region in enumerate(['s1', 's2']):
        ax_curr = ax[i_r]
        
        mean_score_arr, err_score_arr = np.zeros(n_tt), np.zeros(n_tt)
        color_list, label_list = [], []
        i_tt = 0
        for sess_type, tt_test_list in dict_sess_type_tt.items():
            for tt in tt_test_list:
                curr_scores = scores_dict[region][sess_type][tt]
                mean_score_arr[i_tt] = np.mean(curr_scores)
                err_score_arr[i_tt] = np.std(curr_scores) / np.sqrt(len(curr_scores)) * 1.96
                color_list.append(colour_tt_dict[tt])
                label_list.append(label_tt_dict[tt])
                i_tt += 1
                
        ax_curr.bar(x=np.arange(n_tt), height=mean_score_arr - 0.5, yerr=err_score_arr,
                    color=color_list, edgecolor='k', linewidth=2, tick_label=label_list,
                    width=0.8, bottom=0.5)
        ax_curr.set_ylim([0, 1])
        ax_curr.set_xlabel('Trial type')
        ax_curr.set_ylabel('Decoding accuracy')
        ax_curr.text(x=-0.5, y=0.05, s=f'{region.upper()}', fontdict={'weight': 'bold'})
        despine(ax_curr)
        ax_curr.set_xticklabels(ax_curr.get_xticklabels(), rotation=45)
    ax[0].text(y=1.15, x=6, fontdict={'weight': 'bold', 'ha': 'center'},
                s=f'Full population LDA-decoding of trial types vs sham across 6 sessions')
        
def plot_pca_time_aggr_activity(Ses, trial_type_list=['whisker', 'sensory', 'random', 'sham'],
                                merge_trial_types_during_pca=True, verbose=0,
                                plot_ci=True, plot_indiv_trials=False, plot_loadings=True,
                                ax=None, ax_bottom=None, region='s2', n_pcs=3,
                                t_min=-1, t_max=6, save_fig=False):
    if ax is None:
        if plot_loadings:
            fig = plt.figure(constrained_layout=False, figsize=(8, 9))
            gs_top = fig.add_gridspec(ncols=2, nrows=2, wspace=0.6, hspace=0.4, top=0.9, left=0.05, right=0.95, bottom=0.35)
            gs_bottom = fig.add_gridspec(ncols=3, nrows=1, wspace=0.6, top=0.22, left=0.05, right=0.95, bottom=0.05)
            ax = np.array([[fig.add_subplot(gs_top[0, 0]), fig.add_subplot(gs_top[0, 1])], 
                             [fig.add_subplot(gs_top[1, 0]), fig.add_subplot(gs_top[1, 1])]])
            ax_bottom = [fig.add_subplot(gs_bottom[x]) for x in range(3)]
        else:
            fig, ax = plt.subplots(2, 2, figsize=(8, 8), gridspec_kw={'wspace': 0.6, 'hspace': 0.6})
    pc_activity_dict = {}

    ## PCA calculation:
    if merge_trial_types_during_pca:
        if trial_type_list != sorted(trial_type_list):
            print('WARNING: trial type list should be sorted alphabetically. doing that now for you')
            trial_type_list = sorted(trial_type_list)
        selected_ds = Ses.dataset_selector(region=region, min_t=t_min, max_t=t_max,
                                        sort_neurons=False, remove_added_dimensions=True,
                                        trial_type_list=trial_type_list)  # all trial types
        selected_ds_av = selected_ds.groupby('trial_type').mean('trial')  # mean across trials per trial type
        assert selected_ds_av.activity.ndim == 3
        assert list(selected_ds_av.coords.keys()) == ['neuron', 'time', 'trial_type']
        assert (selected_ds_av.trial_type == trial_type_list).all(), f'Order trial types not correct. in Ds: {selected_ds_av.trial_type}, in arg: {trial_type_list}'  ## double check that order of tts is same as input arg, because xarray will sort trial types alphabetically after groupby. (Though technically it doesnt matter because of list concat by trial_types_list order in lines below)
        n_timepoints_per_trial = len(selected_ds_av.time)

        activity_fit = np.concatenate([selected_ds_av.activity.sel(trial_type=tt).data for tt in trial_type_list], axis=1)  # concat average data per trial along time axis
        for i_tt, tt in enumerate(trial_type_list):
            assert (selected_ds_av.activity.sel(trial_type=tt).data == activity_fit[:, (i_tt * n_timepoints_per_trial):((i_tt + 1) * n_timepoints_per_trial)]).all()
        assert activity_fit.ndim == 2
        assert activity_fit.shape[1] == len(trial_type_list) * n_timepoints_per_trial  # neurons x times
        activity_fit = activity_fit.transpose()
        pca = sklearn.decomposition.PCA(n_components=n_pcs)
        pc_activity = pca.fit_transform(X=activity_fit)
        pc_activity = pc_activity.transpose()
        assert pc_activity.shape[1] == len(trial_type_list) * n_timepoints_per_trial
        expl_var = pca.explained_variance_ratio_
        if verbose > 0:
            print(f'Total var expl of all trial types: {expl_var}')

        for i_tt, tt in enumerate(trial_type_list):  ## (we know order is same because of earlier assert)
            pc_activity_dict[tt] = pc_activity[:, (i_tt * n_timepoints_per_trial):((i_tt + 1) * n_timepoints_per_trial)]

        if plot_ci:
            n_trials_per_tt = 100
            assert len(selected_ds.trial) == int(n_trials_per_tt * len(trial_type_list))
            pc_activity_indiv_trials_dict = {tt: np.zeros((n_pcs, n_timepoints_per_trial, n_trials_per_tt)) for tt in trial_type_list}
            assert pca.components_.shape == (n_pcs, len(selected_ds.neuron))
            for i_tt, tt in enumerate(trial_type_list):
                current_tt_ds = selected_ds.where(selected_ds.trial_type == tt, drop=True)
                pca_score_av = pca.score(X=current_tt_ds.mean('trial').activity.data.transpose())
                if verbose > 1:
                    print(f'PCA score of {tt} = {pca_score_av}')
                for i_trial in range(n_trials_per_tt):
                    pca_score_curr_trial = pca.score(current_tt_ds.activity.isel(trial=i_trial).data.transpose())
                    if verbose > 1:
                        print(pca_score_curr_trial)
                    pc_activity_indiv_trials_dict[tt][:, :, i_trial] = np.dot(pca.components_, current_tt_ds.activity.isel(trial=i_trial))


    else:
        assert False, 'This is not the correct of doing this PCA analysis, hence i stopped improving it'
        for i_tt, tt in enumerate(trial_type_list):
            selected_ds = Ses.dataset_selector(region=region, min_t=t_min, max_t=t_max,
                                        sort_neurons=False, remove_added_dimensions=True,
                                        trial_type_list=[tt])  # select just this trial type (tt)
            selected_ds_av = selected_ds.mean('trial')  # trial average activity

            n_timepoints_per_trial = len(selected_ds_av.time)
            pca = sklearn.decomposition.PCA(n_components=n_pcs)
            activity_fit = selected_ds_av.activity.data  # neurons x times
            assert activity_fit.shape[1] == n_timepoints_per_trial
            activity_fit = activity_fit.transpose()
            pc_activity = pca.fit_transform(X=activity_fit)
            pc_activity_dict[tt] = pc_activity.transpose()
            expl_var = pca.explained_variance_ratio_
            if verbose > 0:
                print(tt, 'explained var: ', expl_var)

    ## Plotting
    for i_tt, tt in enumerate(trial_type_list):
            
        ax[0, 0].plot(pc_activity_dict[tt][0, :], pc_activity_dict[tt][1, :], marker='o',
                        color=colour_tt_dict[tt], linewidth=2, label=tt, linestyle='-')

        if plot_indiv_trials:
            indiv_activity_time_av = pc_activity_indiv_trials_dict[tt][:, 45:75, :].mean(1)  # pcs x time points x trials
            ax[0, 0].scatter(indiv_activity_time_av[0, :], indiv_activity_time_av[1, :],
                             marker='x', c=colour_tt_dict[tt])

        i_row = 0
        i_col = 1
        for i_plot in range(n_pcs):
            pc_num = i_plot + 1
            curr_ax = ax[i_row, i_col]
        
            curr_ax.plot(selected_ds_av.time, pc_activity_dict[tt][i_plot, :],
                            color=colour_tt_dict[tt], linewidth=2, label=f'EV {str(np.round(expl_var[i_plot], 3))}')

            if plot_ci:
                indiv_activity = pc_activity_indiv_trials_dict[tt][i_plot, :, :]
                ci = np.std(indiv_activity, 1) * 1.96 / np.sqrt(indiv_activity.shape[1])
                curr_ax.fill_between(selected_ds_av.time, pc_activity_dict[tt][i_plot, :] - ci, 
                                    pc_activity_dict[tt][i_plot, :] + ci, color=colour_tt_dict[tt], alpha=0.5)

            i_col += 1
            if i_col == 2:
                i_col = 0
                i_row += 1

            if plot_loadings:
                ax_bottom[i_plot].hist(pca.components_[i_plot], bins=30, color='grey', linewidth=1)
                ax_bottom[i_plot].set_xlabel(f'PC {pc_num} loading')
                ax_bottom[i_plot].set_ylabel('Frequency')
                ax_bottom[i_plot].set_title(f'Loadings of PC {pc_num}', fontdict={'weight': 'bold'})

    ## Cosmetics:
    despine(ax[0, 0])
    ax[0, 0].set_xlabel('PC 1')
    ax[0, 0].set_ylabel('PC 2')
    ax[0, 0].legend(loc='best', frameon=False)
    ax[0, 0].set_title('State space PC 1 vs PC 2', fontdict={'weight': 'bold'})

    i_row = 0
    i_col = 1
    for i_plot in range(n_pcs):
        pc_num = i_plot + 1
        curr_ax = ax[i_row, i_col]
        despine(curr_ax)
        curr_ax.set_xlabel('Time (s)')
        curr_ax.set_ylabel(f'PC {pc_num}')
        if merge_trial_types_during_pca:
            curr_ax.set_title(f'PC {pc_num} activity, {int(np.round(expl_var[i_plot] * 100))}% EV of trial-av.', fontdict={'weight': 'bold'})
        else:
            curr_ax.set_title(f'PC {pc_num} activity', fontdict={'weight': 'bold'})
            curr_ax.legend(loc='best')
        i_col += 1
        if i_col == 2:
            i_col = 0
            i_row += 1
        if plot_loadings:
            despine(ax_bottom[i_plot])
    
    plt.suptitle(f'Trial-average PC traces in {region.upper()} region of {Ses.session_name_readable}', fontdict={'weight': 'bold'})

    if save_fig:
        plt.savefig(f'/home/tplas/repos/Vape/jupyter/thijs/figs/pca_activity__{Ses.sess_type}__{Ses.session_name_readable}_{region.upper()}.pdf', bbox_inches='tight')

def manual_poststim_response_classifier(Ses, region='s2', tt_1='sensory', tt_2='sham',
                                        t_min=1, t_max=2, time_aggr_method='average',
                                        n_shuffles=5000, verbose=1, plot_hist=True, ax=None):    
    tt_list = [tt_1, tt_2]
    assert len(tt_list) == 2, 'multi classification not implemented'

    ## Get data
    ds = Ses.dataset_selector(region=region, remove_added_dimensions=True,
                              min_t=t_min, max_t=t_max,
                            sort_neurons=False, trial_type_list=tt_list)
    n_trials_per_tt = 100
    tmp_data_dict = {}
    time_av_responses_dict = {}
    for i_tt, tt in enumerate(tt_list):
        tmp_data_dict[tt] = ds.activity.where(ds.trial_type == tt, drop=True).mean('neuron').data.transpose()
        if time_aggr_method == 'average':
            time_av_responses_dict[tt] = tmp_data_dict[tt][:, :].mean(1)
        else:
            assert False, 'no other aggretation method than average implemented'
        assert len(time_av_responses_dict[tt]) == 100

    ## Get real classification performance
    mean_response_per_tt_dict = {tt: time_av_responses_dict[tt].mean() for tt in tt_list}
    threshold = 0.5 * (mean_response_per_tt_dict[tt_list[0]] + mean_response_per_tt_dict[tt_list[1]])
    correct_classification_dict = {}
    for tt in tt_list:
        if mean_response_per_tt_dict[tt] > threshold:
            correct_classification_dict[tt] = time_av_responses_dict[tt] > threshold 
        else:
            correct_classification_dict[tt] = time_av_responses_dict[tt] < threshold 
        if verbose > 0:
            print(f'Number of correctly classified trials of {tt}: {np.sum(correct_classification_dict[tt])}')

    ## Shuffled classification performance:
    n_correct_class_shuffled_dict = {tt: np.zeros(n_shuffles) for tt in tt_list}
    all_responses = np.concatenate([time_av_responses_dict[tt] for tt in tt_list], axis=0)
    assert len(all_responses) == len(tt_list) * n_trials_per_tt
    for i_shuf in range(n_shuffles):
        random_trial_inds = np.random.permutation(len(all_responses))
        shuffled_responses_all = all_responses[random_trial_inds]
        shuffled_responses_1 = shuffled_responses_all[:n_trials_per_tt]
        shuffled_responses_2 = shuffled_responses_all[n_trials_per_tt:]
        mean_1, mean_2 = shuffled_responses_1.mean(), shuffled_responses_2.mean()
        threshold = 0.5 * (mean_1 + mean_2)
        if mean_1 > threshold:
            correct_sh_resp_1 = shuffled_responses_1 > threshold
            correct_sh_resp_2 = shuffled_responses_2 < threshold
        else:
            correct_sh_resp_1 = shuffled_responses_1 < threshold
            correct_sh_resp_2 = shuffled_responses_2 > threshold
        n_correct_class_shuffled_dict[tt_list[0]][i_shuf] = np.sum(correct_sh_resp_1)
        n_correct_class_shuffled_dict[tt_list[1]][i_shuf] = np.sum(correct_sh_resp_2)
            
    ## Compute p value using two sided z test
    n_cor_real_dict, p_val_dict = {}, {}
    for tt in tt_list:
        n_cor_real_dict[tt] = np.sum(correct_classification_dict[tt])
        mean_n_cor_sh = np.mean(n_correct_class_shuffled_dict[tt])
        std_n_cor_sh = np.std(n_correct_class_shuffled_dict[tt])
        zscore_n_cor = (n_cor_real_dict[tt] - mean_n_cor_sh) / std_n_cor_sh
        p_val_dict[tt] = scipy.stats.norm.sf(np.abs(zscore_n_cor)) * 2
        if verbose > 0:
            print(f'Two-sided p value of {tt} = {p_val_dict[tt]}')

    ## Plot
    if plot_hist:
        if ax is None:
            ax = plt.subplot(111)
        hist_n ,_, __ = ax.hist([n_correct_class_shuffled_dict[tt_list[x]] for x in range(2)], 
                    bins=np.arange(40, 70, 1), 
                    density=True, label=tt_list, color=[colour_tt_dict[tt] for tt in tt_list])
        ax.set_xlabel('percentage correctly classified trials')
        ax.set_ylabel('PDF')
        ax.set_title(f'Distr. of correctly classified SHUFFLED trials\nN_bootstrap={n_shuffles}, {region.upper()}, {Ses.session_name_readable}')
        despine(ax)
        max_vals = np.max(np.array([np.max(hist_n[x]) for x in range(2)]))
        for i_tt, tt in enumerate(tt_list): 
            ax.text(s=u"\u2193" + f' (P = {np.round(p_val_dict[tt], 3)})', 
                    x=n_cor_real_dict[tt], y=max_vals * (1 + 0.1 * (i_tt + 1)),
                    fontdict={'ha': 'left', 'color': colour_tt_dict[tt], 'weight': 'bold'})
        ax.set_ylim([0, max_vals * 1.3])

def plot_cross_temp_corr(ds, ax=None, name=''):
    n_trials = len(ds.trial)
    tmpcor = np.stack([np.corrcoef(ds.activity.isel(trial=x).data.transpose()) for x in range(n_trials)])
    meancor = tmpcor.mean(0)

    if ax is None:
        ax = plt.subplot(111)
    sns.heatmap(meancor, ax=ax, cmap=sns.color_palette("cubehelix", as_cmap=True), vmax=0.5, vmin=0)
    create_time_axis(ax=ax, time_arr=ds.time.data, axis='x')
    create_time_axis(ax=ax, time_arr=ds.time.data, axis='y')
    ax.invert_yaxis()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Time (s)')
    ax.set_title(f'Mean cross-temporal correlation across {n_trials} {name} trials')


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