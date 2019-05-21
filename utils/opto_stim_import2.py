import warnings
warnings.filterwarnings("ignore")

import numpy as np
import glob
import os
import csv
import sys
sys.path.append('..')
from utils.data_import import Session
#change this
from utils.utils_funcs import paq_data
from utils.utils_funcs import d_prime as pade_dprime
import utils.gsheets_importer as gsheet
from utils.paq2py import paq_read
from utils.rsync_aligner import Rsync_aligner


class OptoStimBasic():

    def __init__(self, txt_path):
        '''
        proceses information from individual pycontrol txt file
        relevant to all opto_stim task flavours
        '''

        self.session = Session(txt_path)
        self.print_lines = self.session.print_lines

        self._outcome_lists()

    def pl_times(self, str_):
        '''returns the time of all print lines with str_ in the line'''
        return [float(line.split(' ')[0]) for line in self.print_lines if str_ in line]

    def _appender(self, str_, outcome):
        '''appends trial outcomes and their times to lists'''
        times = self.pl_times(str_)
        if times:
            self.trial_time.append(times)
            self.outcome.append([outcome]*len(times))

    def _outcome_lists(self):
        '''drives the appender function to build trial outcome lists'''
        self.trial_time = []
        self.outcome = []

        self._appender('earned_reward', 'hit')
        self._appender('missed trial', 'miss')
        self._appender('correct rejection', 'cr')
        self._appender('false positive', 'fp')

        flatten = lambda l: [item for sublist in l for item in sublist]
        self.trial_time = flatten(self.trial_time)
        self.outcome = flatten(self.outcome)

        sort_idx = np.argsort(self.trial_time)
        self.trial_time = np.array(self.trial_time)[sort_idx]
        self.outcome = np.array(self.outcome)[sort_idx]

        self.trial_time, self.outcome = OptoStimBasic._debounce(self.trial_time, self.outcome, 2000)
        self.trial_time = self.test_import_and_slice(self.trial_time)
        self.outcome = self.test_import_and_slice(self.outcome)

    @classmethod
    def _debounce(cls, arr1, arr2, window):
        '''
        function to debounce arrays based on the numeric values of arr1, used for example when the trial outcome is erroneously printed twice
        removes subsequent elements of an array that occur within a time window to the previous element
        '''
        for i,a in enumerate(arr1):
            # finish debouncing if reach the final element
            if i == len(arr1) - 1: return arr1, arr2

            diff = arr1[i+1] - arr1[i]
            if diff <= window:
                arr1 = np.delete(arr1, i+1)
                arr2 = np.delete(arr2, i+1)
                # recursively call debounce function
                OptoStimBasic._debounce(arr1,arr2, window)

    @property
    def n_trials_complete(self):
        return len(self.session.times.get('ITI'))

    @property
    def trial_type(self):
        trial_type = ['go' if t == 'miss' or t == 'hit' else 'nogo' for t in self.outcome]
        return self.test_import_and_slice(trial_type)

    @property
    def trial_start(self):
        go_start = self.session.times.get('detect_lick_go')
        nogo_start = self.session.times.get('detect_lick_nogo')
        trial_start = np.sort(np.hstack((go_start, nogo_start)))

        return self.test_import_and_slice(trial_start)
        #trial_start = np.sort(go_start+nogo_start)
        #return self.test_import_and_slice(trial_start)
        #     ''' the time that the trial started scope and normal task print different strings at trial start '''
        #     trial_start = [float(line.split(' ')[0]) for line in self.print_lines if 'goTrial' in line or 'nogo_trial' in line or 'Trigger SLM trial' in line or 'Start NOGO trial' in line]
        #     return self.test_import_and_slice(trial_start)

    @property
    def online_dprime(self):
        return [float(line.split(' ')[3]) for line in self.print_lines if 'd_prime is' in line]

    @property
    def binned_licks(self):
        ''' gets the lick times normalised to the start of each trial '''
        licks = self.session.times.get('lick_1')

        binned_licks = []
        for i, t_start in enumerate(self.trial_start):

            # arbitrary big number to prevent index error on last trial
            if i == len(self.trial_start)-1: t_end = 10e100
            else: t_end = self.trial_start[i+1]

            #find the licks occuring in each trial
            trial_idx = np.where((licks>=t_start) & (licks<=t_end))[0]

            #normalise to time of trial start
            trial_licks = licks[trial_idx] - t_start

            binned_licks.append(trial_licks)

        return self.test_import_and_slice(binned_licks)



    @property
    def hit_rate(self):
        n_hits = (self.outcome=='hit').sum()
        n_miss = (self.outcome=='miss').sum()
        return n_hits / (n_hits + n_miss)

    @property
    def fp_rate(self):
        n_fa = (self.outcome=='fp').sum()
        n_cr = (self.outcome=='cr').sum()
        return n_fa / (n_fa + n_cr)

    @property
    def dprime(self):
        #use matthias approximation
        return pade_dprime(self.hit_rate, self.fp_rate)

    def test_import_and_slice(self, list_):
        '''
        test that import has been done correctly and slice to remove end trial if IndexError in task
        tests that list_if of correct length and slices to only include full trial_start
        args = list to test for correct length and slice
        '''

        len_test = lambda x : True if self.n_trials_complete <= x <= self.n_trials_complete + 1 else False

        assert len_test(len(list_)), 'error importing, list of wrong length'

        return list_[0:self.n_trials_complete]



class OptoStim1p(OptoStimBasic):
        def __init__(self, txt_path):

            '''init this class to process the 1p opto_stim txt file in txt_path'''

            super().__init__(txt_path)
            self.autoreward()


        @property
        def LED_current(self):
            '''gets the LED current on each trial'''

            if self.session.task_name == 'opto_stim':
                LED_current = [float(line.split(' ')[4]) for line in self.print_lines if 'LED current is' in line and 'now' not in line]
            elif self.session.task_name == 'opto_stim_psychometric':
                LED_current =  [float(line.split(' ')[3]) for line in self.print_lines if 'LED_current is' in line]
            else:
                raise ValueError('task not recognised')

            return self.test_import_and_slice(LED_current)

        def autoreward(self):
            '''detect if pycontrol went into autoreward state '''

            autoreward_times = self.session.times.get('auto_reward')

            self.autorewarded_trial = []
            for i,t_start in enumerate(self.trial_time):
                #arbitrary big number to prevent index error on last trial

                if i == self.n_trials_complete-1: t_end = 10e100
                else: t_end = self.trial_time[i+1]

                is_autoreward = next((a for a in autoreward_times if a >= t_start and a < t_end), False)
                if is_autoreward: self.autorewarded_trial.append(True)
                else: self.autorewarded_trial.append(False)

            ## did mouse get out of autreward phase?
            time_autoswitch = self.pl_times('switching out of auto')[0]

            ## find the trial number this happened by finding the index of the closest trial start time (+1)
            if time_autoswitch:
                self.trial_autoswitch = np.abs(self.trial_time - time_autoswitch).argmin() + 1
            else:
                self.trial_autoswitch = None


class OptoStim2p(OptoStimBasic):
    def __init__(self, txt_path):
        '''init this class to process the 2p opto_stim txt file in txt_path'''

        super().__init__(txt_path)

        #the line that triggers an SLM trial throuh blimp
        _trigger_lines = [line for line in self.print_lines if 'Trigger SLM trial Number' in line]

        self.barcode = [float(line.split(' ')[7]) for line in _trigger_lines]
        self.SLM_trial_number = [float(line.split(' ')[5]) for line in _trigger_lines]

        assert len(self.barcode) == self.trial_type.count('go') == max(self.SLM_trial_number)
        self.rsync = self.session.times.get('rsync')


class BlimpImport(OptoStim2p):

    #from the URL
    sheet_ID = '1GG5Y0yCEw_h5dMfHBnBRTh5NXgF6NqC_VVGfSpCHroc'
    packerstation_path = '/home/jamesrowland/Documents/packerstation/jrowland/Data'

    #the name of the file path column
    date_header = 'Date'
    t_series_header = 't-series name'
    paq_header = '.paq file'
    naparm_header = 'Naparm'
    blimp_header = 'blimp folder'
    pycontrol_header = 'pycontrol txt'

    #column telling whether to analyse or not
    analyse_bool_header = 'Analyse'

    def __init__(self, mouse_id):
        '''
        class to import 2-photon blimp experiments based on the
        gsheet given in sheet ID
        '''
        self.mouse_id = mouse_id

        self.parse_spreadsheet()


    @property
    def df(self):
        sheet_name = self.mouse_id + '!A1:Z69'
        df = gsheet.gsheet2df(BlimpImport.sheet_ID, HEADER_ROW=2, SHEET_NAME=sheet_name)
        #comment this out if not using behaviour spreadsheet
        df = gsheet.correct_behaviour_df(df)
        return df

    def parse_spreadsheet(self):

        idx_analyse = gsheet.df_bool(self.df, BlimpImport.analyse_bool_header)
        idx_2p = gsheet.df_bool(self.df, 'Trained 2P')
        idx_1p = gsheet.df_bool(self.df, 'Trained 1P')

        intersect = lambda A, B: set(A) - (set(A) - set(B))

        rows_2p = intersect(idx_analyse, idx_2p)
        rows_1p = intersect(idx_analyse, idx_1p)

        self.dates_2p = gsheet.df_col(self.df, BlimpImport.date_header, rows_2p)
        self.paqs = gsheet.df_col(self.df, BlimpImport.paq_header, rows_2p)
        self.naparm_folders = gsheet.df_col(self.df, BlimpImport.naparm_header, rows_2p)
        self.blimp_folders = gsheet.df_col(self.df, BlimpImport.blimp_header, rows_2p)
        self.pycontrol_folders = gsheet.df_col(self.df, BlimpImport.pycontrol_header, rows_2p)

        assert len(self.paqs) == len(self.naparm_folders) == len(self.blimp_folders) == len(self.dates_2p)
        num_runs = len(self.paqs)

    def get_object_and_test(self, run):
        '''build obejct for a sepcific run and test that details have been entered correctly'''

        date = self.dates_2p[run]
        paq = self.paqs[run]
        naparm = self.naparm_folders[run]
        blimp = self.blimp_folders[run]
        pycontrol = self.pycontrol_folders[run]

        umbrella = os.path.join(BlimpImport.packerstation_path, date)
        self.blimp_path, self.naparm_path = gsheet.path_finder(umbrella, blimp, naparm, is_folder=True)
        self.pycontrol_path, self.paq_path = gsheet.path_finder(umbrella, pycontrol, paq, is_folder=False)

        with open(os.path.join(self.blimp_path, 'blimpAlignment.txt'), 'r') as f:
            barcode = [float(line.split(' ')[7][:-1]) for line in f.readlines()] #take the full stop off the end

        #build behaviour object from the pycontrol txt file
        self.behav = OptoStim2p(self.pycontrol_path)

        #test that the list of barcodes printed in the pycontrol sequence is contained in the
        #list of barcodes from the alignment folder
        if not ''.join([str(b) for b in self.behav.barcode]) in ''.join([str(b) for b in barcode]):
            raise ValueError('pycontrol {} does not match blimp folder {}'.format(pycontrol, blimp))
        else:
            print('pycontrol {} successfully matched to blimp folder {}'.format(pycontrol, blimp))

        paq_obj = paq_read(self.paq_path)

        paq_rsync = paq_data(paq_obj, 'pycontrol_rsync', threshold_ttl=True, plot=False)

        try:
            aligner = Rsync_aligner(pulse_times_A=self.behav.rsync, pulse_times_B=paq_rsync,
                                    units_B=1000/paq_obj['rate'],  plot=False, raise_exception=True)

            print('pycontrol {} rsync successfully matched to paq {}'.format(pycontrol, paq))
        except:
            print('pycontrol {} rsync could not be matched to paq {}'.format(pycontrol, paq))

        return self.behav
