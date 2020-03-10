import numpy as np
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from subsets_analysis import Subsets
import utils_funcs as utils
from scipy import signal
import random
''' functions that operate on BlimpImport objects (runs) '''


def filter_unhealthy_cells(run, threshold=5):

    max_cell = np.max(run.flu, 1)
    min_cell = np.min(run.flu, 1)
    healthy = np.where((max_cell < threshold) & (min_cell > -threshold))[0]
    run.flu = run.flu[healthy, :]
    run.frames_ms = run.frames_ms[healthy, :]
    run.frames_ms_pre = run.frames_ms_pre[healthy, :]
    run.stat = run.stat[healthy]
    run.spks = run.spks[healthy, :]
    return run


def select_s2(run):
    '''n.b. currently only works for three plane imaging sessions'''

    s2_cells = []
    for i, s in enumerate(run.stat):
        if s['iplane'] == 0 and s['med'][1] > 512:
            s2_cells.append(i)
        elif s['iplane'] == 1 and s['med'][1] > 1536:
            s2_cells.append(i)
        elif s['iplane'] == 2 and s['med'][1] > 512:
            s2_cells.append(i)

    run.flu = run.flu[s2_cells, :]
    run.frames_ms = run.frames_ms[s2_cells, :]
    run.frames_ms_pre = run.frames_ms_pre[s2_cells, :]
    run.stat = run.stat[s2_cells]
    run.spks = run.spks[s2_cells, :]

    return run


def filter_trials(run, good=True, dp_thresh=1.5, window_size=5, plot=False):
    ''' Takes a run object and calculates a running d-prime
        throughout the trials, returns a list of trial idxs
        where performance was above or below this window

        Inputs:
        run -- BlimpImport object
        good -- whether to return trial idxs where the animal
                if performing good or bad
        window_size -- size (n_trials) of the running dprime window
        plot -- whether to plot the running dprime window

        Returns:
        idxs of good or bad trials

        '''

    subsets = Subsets(run)
    n_trials = len(run.outcome)

    easy_idx = np.where(subsets.trial_subsets == 150)[0]
    nogo_idx = np.where(subsets.trial_subsets == 0)[0]

    go_outcome = []
    nogo_outcome = []

    for trial in run.outcome[easy_idx]:
        if trial == 'hit':
            go_outcome.append(1)
        elif trial == 'miss':
            go_outcome.append(0)
        else:
            raise ValueError

    for trial in run.outcome[nogo_idx]:
        if trial == 'fp':
            nogo_outcome.append(1)
        elif trial == 'cr':
            nogo_outcome.append(0)
        else:
            raise ValueError

    running_go = np.convolve(go_outcome, np.ones(
        (window_size,))/window_size, mode='same')
    running_nogo = np.convolve(nogo_outcome, np.ones(
        (window_size,))/window_size, mode='same')

    running_go = np.interp(np.arange(n_trials), easy_idx, running_go)
    running_nogo = np.interp(np.arange(n_trials), nogo_idx, running_nogo)


    # resampling can give < 1 or > 0
    cap = lambda lst: [max(min(x, 1), 0) for x in lst] 
    running_go = cap(running_go)
    running_nogo = cap(running_nogo)

    running_dp = [utils.d_prime(go, nogo)
                  for go, nogo in zip(running_go, running_nogo)]

    running_dp = np.array(running_dp)

    marker_map = {
                'hit': 'red',
                'miss': 'blue',
                'cr': 'blue',
                'fp':'red'
                }

    trial_marker = [marker_map[i] for i in run.outcome]
    

    if plot:
        plt.figure(figsize=(15,15))
        plt.plot(running_go, color='red', label='Running Hit Rate')
        plt.plot(running_nogo, color='blue', label='Running FA rate')
        plt.plot(running_dp, color= 'green', label=' Running dprime')
        plt.xlabel('Trial Number')
        for trial, outcome in enumerate(run.outcome):
            if outcome == 'hit' or outcome == 'miss':
                y_pos = 1
            else:
                y_pos = 0
            plt.plot(trial, y_pos, marker='o', color=trial_marker[trial], markersize=8)

        plt.annotate('Go Trial', [n_trials+5, 0.97], fontsize=15)
        plt.annotate('NoGo Trial', [n_trials+5, -0.03], fontsize=15)
        plt.legend(fontsize=10)

    if good:
        return [True if rdp > dp_thresh else False for rdp in running_dp]
    else:
        return [True if rdp < dp_thresh else False for rdp in running_dp]


def get_spont_trials(run, pre_frames=5, post_frames=9, n_trials=10):
    ''' Creates pseudo-trials from a run object where no 
        behaviour is occuring

        Inputs:
        run -- BlimpImport object
        pre_frames -- the number of frames to include before
                      trial start
        post_frames -- the number of frames to inlcude after
                       trial start
        n_trials -- the number of trials to generate

        Returns:
        spontaneous array [n_cells x n_trials x n_frames]

        '''

    # first non-nan frame signifying start of behaviour
    # (-1 in case different cells)
    idx_first = min(np.where(~np.isnan(run.frames_ms[0, :]))[0]) - 1

    # frames that were not in the prereward phase
    non_pre = np.where(np.isnan(run.frames_ms_pre[0, :]))[0]

    assert max(non_pre) == run.flu.shape[1] - 1
    spont_frames = non_pre[non_pre < idx_first]

    spont_flu = run.flu[:, spont_frames]

    # make a tbt array of 10 spontaneous flu 'trials'
    len_trial = pre_frames + post_frames

    spont_trials = []

    for i in range(n_trials):
        t_start = random.choice(
            np.arange(len_trial, spont_flu.shape[1] - len_trial))
        trial = spont_flu[:, t_start:t_start+len_trial]
        spont_trials.append(trial)

    return np.swapaxes(np.array(spont_trials), 0, 1)


def get_time_to_lick(run, fs=5):

    time_to_lick = []

    # inter_frame_interval in ms
    ifi_ms = 1000 / fs
    for b in run.binned_licks_easytest:
        if len(b) == 0:
            time_to_lick.append(np.nan)
        else:
            time_to_lick.append(math.floor(b[0] / ifi_ms))

    return np.array(time_to_lick)


def subtract_kernal(flu_array, run, trials_to_sub='all',
                    offset_sub=None, pre_frames=5, post_frames=9, plot=False):

    if trials_to_sub == 'all':
        trials_to_sub = np.arange(flu_array.shape[1])

    if offset_sub is None:
        offset_sub = np.repeat(0, flu_array.shape[1])

    assert flu_array.shape[1] == len(offset_sub)

    # get lick kernal for all time_to_lick offsets
    kernals = []
    for offset in np.arange(np.max(offset_sub[trials_to_sub])+1):
        offset = int(offset)
        kernals.append(utils.build_flu_array(run, run.pre_reward,
                                             pre_frames+offset, post_frames-offset, True))

    # mean across trials [n_cells x pre_frames+post_frames]
    mean_kernals = [np.nanmean(kernal, 1) for kernal in kernals]
    # type error with nans here means trials are misaligned
    offset_sub[trials_to_sub]

    pre_post = np.nanmean(mean_kernals[0][:, pre_frames:post_frames],
                          1) - np.nanmean(mean_kernals[0][:, 0:pre_frames], 1)
    cell_licky_idx = np.flip(np.argsort(pre_post))

    if plot:
        fig = plt.figure(figsize=(22, 8))
        x_axis = np.arange(pre_frames+post_frames)
        for i, cell in enumerate(cell_licky_idx[:12]):
            plt.subplot(3, 4, i+1)
            plt.plot(x_axis, np.nanmean(
                flu_array[cell, :, :], 0), 'blue', label='Mean Hit Trials')
            plt.plot(x_axis, mean_kernals[0][cell, :],
                     'green', label='Lick Kernal')
            plt.plot(x_axis, np.nanmean(
                flu_array[cell, :, :], 0) - mean_kernals[offset][cell, :], 'pink', label='Kernal subtracted')
            plt.xticks(x_axis)
        handles, labels = plt.gca().get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center')

    for i, t in enumerate(trials_to_sub):
        offset = int(offset_sub[t])
        mean_kernal = mean_kernals[offset]

        flu_array[:, t, :] = flu_array[:, t, :] - mean_kernal

    return flu_array, cell_licky_idx, pre_post


def subsets_diff_plotter(runs, behaviour_list, pre_frames=5, post_frames=9,
                         offset=4, good=True, is_spks=False):

    subset_sizes = np.unique(Subsets(runs[0]).trial_subsets)

    hit_diffs = []
    miss_diffs = []

    # remove 0 and 150
    subset_sizes = subset_sizes[1:-1]

    for i, sub in enumerate(subset_sizes):

        hit_trials = [utils.intersect(np.where((run.outcome == 'hit')
                                               & (Subsets(run).trial_subsets == sub))[0],
                                      filter_trials(run, good=good))
                      for run in runs]

        miss_trials = [utils.intersect(np.where((run.outcome == 'miss')
                                        & (Subsets(run).trial_subsets == sub))[0],
                                       filter_trials(run, good=good))
                       for run in runs]

        hit_diff = utils.prepost_diff(behaviour_list, pre_frames=pre_frames,
                                      post_frames=post_frames, offset=offset,
                                      filter_list=hit_trials)

        miss_diff = utils.prepost_diff(behaviour_list, pre_frames=pre_frames,
                                       post_frames=post_frames, offset=offset,
                                       filter_list=miss_trials)

        if i == 0:
            plt.bar(i-0.1, np.nanmean(hit_diff),
                    color=sns.color_palette()[0], label='Hit Trials')
            plt.bar(i, np.nanmean(miss_diff), color=sns.color_palette()
                    [6], label='Miss Trials')
        else:
            plt.bar(i-0.1, np.nanmean(hit_diff), color=sns.color_palette()[0])
            plt.bar(i, np.nanmean(miss_diff), color=sns.color_palette()[6])

        hit_diffs.append(hit_diff)
        miss_diffs.append(miss_diff)

    plt.legend()

    if is_spks:
        plt.ylabel(r'Mean spks Jump Post-Pre')
    else:
        plt.ylabel('Mean $\Delta $F/F Jump Post-Pre')

    plt.xlabel('Number of cells stimulated')
    sns.despine()
    plt.xticks(range(len(subset_sizes)), subset_sizes)

    return np.array(hit_diffs), np.array(miss_diffs)


def raw_data_plotter(run, unit=0, combined=True):

    if combined:
        s2p_path = os.path.join(run.s2p_path, 'combined')
    else:
        s2p_path = os.path.join(run.s2p_path, 'plane0')
        
    neuropil = np.load(os.path.join(s2p_path, 'Fneu.npy')) 

    unit = unit
    fig, ax1 = plt.subplots(figsize=(15, 10))

    praw = ax1.plot(run.flu_raw[unit, :], label='Flu Raw',
                    color=sns.color_palette()[0])
    pn = ax1.plot(neuropil[unit, :], label='Neuropil',
                  color=sns.color_palette()[1])
    ds = ax1.plot(run.spks[unit, :], label='Deconvolved Spikes',
                  color=sns.color_palette()[2])
    # ax1.set_xlim((4000, 5000))
    ax1.set_ylim((-100, 3500))
    ax1.set_ylabel('Raw Data')
    ax1.legend(fontsize=10)

    ax2 = ax1.twinx()
    df = ax2.plot(run.flu[unit, :], 
                  label='$\Delta $F/F Neuropil Subtracted',
                  color=sns.color_palette()[4])
    ax2.set_ylim((-2, 3.5))
    # ax2.set_xlim((5000, 7000))
    ax2.set_ylabel('$\Delta $F/F')

    lns = praw+pn+df+ds
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)


def get_bad_frames(run, save_paths, fs=30):    

    pre_frames = math.ceil(1*fs)
    post_frames = math.ceil(1.5*fs)

    paqio_frames = utils.tseries_finder(run.num_frames, run.frame_clock)
    trial_start = utils.get_spiral_start(run.x_galvo_uncaging, run.paq_rate*6)

    bad_frames = []
    trial_starts = []
    for i, start in enumerate(trial_start):
        
        frames, start_idx = utils.get_trial_frames(paqio_frames, start, 
                                                   pre_frames, post_frames, 
                                                   paq_rate=run.paq_rate, 
                                                   fs=fs)
        trial_starts.append(start_idx)
        bad_frames.append(frames)

    flattened = [bf for bf in bad_frames if bf is not None]
    flattened = [item for sublist in flattened for item in sublist]

    for save_path in save_paths:
        np.save(os.path.join(save_path, 'bad_frames.npy'), flattened, allow_pickle=True)
        print('Bad frames calculated and saved to {}'.format(save_path))

    return trial_starts, bad_frames

def get_run_number(run):
    ''' Get the number of a run object'''
    df = run.df
    df1 = df[df['pycontrol txt'].str.contains(run.run_pycontrol_txt)] 
    return int(df1['Run Number'].values[0])

def manual_correct(run):

    ''' Carefully tested manual correction to fix e.g. when blimp
        is restarted and paq has erroneous trial starts
        '''

    run_number = get_run_number(run)

    if run.mouse_id == 'J063' and run_number==10 and\
    len(run.spiral_start) - 2 == len(run.trial_start):
        # The first two trials here were from a stopped and started pyc file
        run.spiral_start = run.spiral_start[2:]

    if len(run.spiral_start) == len(run.trial_start) - 1:
        run.spiral_start = np.append(run.spiral_start, np.nan)

    elif run.mouse_id == 'RL072' and run_number==20 and\
    len(run.spiral_start) == len(run.trial_start)+1:
        run.spiral_start = run.spiral_start[1:]
        # The galvo was parked at 1 volt for some reason during
        # prereward, manually correct this
        run.x_galvo_uncaging[0:int(0.145088e8)] = -1

    elif run.mouse_id == 'RL072' and run_number==24 and\
    len(run.spiral_start) == len(run.trial_start) + 1:
        run.spiral_start = run.spiral_start[:-1]
        # Galvo again parked in a weird place before session started
        run.x_galvo_uncaging[:int(1.610905e7)] = -1

    return run


def spiral_tstart(run):

    '''
    Get the times that a run's trials started based on
    spiralling of the x-galvo
    '''

    run.spiral_start = utils.get_spiral_start(run.x_galvo_uncaging, run.paq_rate * 3)

    # Closely tested manual corrections for the x_galvo
    # to deal with e.g. restarted pycontrol sessions
    run = manual_correct(run)
    
    try:
        jitter_tstart(run)
    except ValueError:
        print(run.mouse_id)
        print('x_galvo is length {} trial start is length {}'.
              format(len(run.spiral_start), len(trial_start)))
        raise

    return run

def jitter_tstart(run):
    ''' requires spiral_start '''
    return run.aligner.B_to_A(run.spiral_start) -  run.trial_start


def tanke_exclusion(run):
    
    trial_subsets = Subsets(run).trial_subsets
    assert len(trial_subsets) == len(run.outcome)
    
    easy_idxs = np.where(trial_subsets==150)[0]
    # Only including 40 and 50 for now, change going forward
    #test_idxs = np.where((trial_subsets!=0) & (trial_subsets!=150))[0]
    test_idxs = np.where((trial_subsets==40) | (trial_subsets==50))[0]
    nogo_idxs = np.where(trial_subsets==0)[0] 
    easy_outcomes = run.outcome[easy_idxs]
    
    test_include = utils.between_two_hits(test_idxs, easy_idxs, easy_outcomes)
    nogo_include = utils.between_two_hits(nogo_idxs, easy_idxs, easy_outcomes)
    
    return test_idxs[test_include], nogo_idxs[nogo_include]


def dprime_tanke(run):

    test_idxs, nogo_idxs = tanke_exclusion(run)
    hit_rate = (run.outcome[test_idxs] == 'hit').sum() / len(test_idxs)
    fp_rate = (run.outcome[nogo_idxs] == 'fp').sum() / len(nogo_idxs)
    test_dp = utils.d_prime(hit_rate, fp_rate)

    return test_dp




