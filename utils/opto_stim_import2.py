import numpy as np
import glob
import os
import csv
import sys
sys.path.append('..')
from utils.data_import import Session
from utils.utils_funcs import d_prime as pade_dprime
from utils.gsheets_importer import gsheet2df, correct_behaviour_df

class OptoStimTxt():

    def __init__(self, txt_path):
        '''
        proceses individual opto_stim pycontrol txt file
        '''

        self._session = Session(txt_path)
        self._print_lines = self._session.print_lines

        self._outcome_lists()


    def _pl_times(self, str_):
        '''returns the time of all print lines with str_ in the line'''
        return [float(line.split(' ')[0]) for line in self._print_lines if str_ in line]

    def _appender(self, str_, outcome):
        '''appends trial outcomes and their times to lists'''
        times = self._pl_times(str_)
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

        self.trial_time, self.outcome = OptoStimTxt._debounce(self.trial_time, self.outcome, 2000)

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
                del arr1[i+1]
                del arr2[i+1]
                # recursively call debounce function
                self.debounce(arr1,arr2, window)

    @property
    def trial_type(self):
        return ['go' if t == 'miss' or t == 'hit' else 'nogo' for t in self.outcome]

    @property
    def trial_start(self):
        ''' the time that the trial started scope and normal task print different strings at trial start '''
        return [float(line.split(' ')[0]) for line in self._print_lines if 'goTrial' in line or 'nogo_trial' in line or 'Start SLM trial' in line or 'Start NOGO trial' in line]

    @property
    def online_dprime(self):
        return [float(line.split(' ')[3]) for line in self.print_lines if 'd_prime is' in line]

    @property
    def binned_licks(self):
        ''' gets the lick times normalised to the start of each trial '''
        licks = self._session.times.get('lick_1')

        binned_licks = []
        for i, t_start in enumerate(self.trial_start):

            # arbitrary big number to prevent index error on last trial
            if i == len(self.trial_start)-1: t_end = 10e100
            else: t_end = self.trial_start[i+1]

            #find the licks occuring in each trial
            trial_IND = np.where((licks>t_start) & (licks<t_end))[0]

            #normalise to time of trial start
            trial_licks = licks[trial_IND] - t_start

            binned_licks.append(trial_licks)

        return binned_licks

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















class OptoStimImport():

    #ID of the Optistim Behaviour gsheet (from the URL)
    sheet_ID = '1GG5Y0yCEw_h5dMfHBnBRTh5NXgF6NqC_VVGfSpCHroc'

    def __init__(self, ID):

        '''
        common class to parse data from all opto_stim behaviour tasks
        to a pandas dataFrame.
        Takes the google sheet addesss in sheet_ID and creates a pandas
        dataframe with metadata and 1p info from task

        Inputs
        ID: The ID of the mouse, need to match the spreadsheet name
        '''

        self.ID = ID

        @property
        def df(self):
            self.df = gsheet2df(OptoStimImport.sheet_ID, 2, SHEET_NAME=self.ID)
            self.df = correct_behaviour_df(self.df)

        def split_df(self, col_id):
            '''slice dataframe by boolean value of col_id'''
            idx = self.df.index[self.df[col_id]=='TRUE']
            return self.df.loc[idx, :]


if __name__ == "__main__":
    #importer = OptoStimImport('RL019')
    tp = OptoStimTxt('/home/jamesrowland/Documents/packerstation/jrowland/Data/2019-04-19/pycontrol_data/RL019-2019-04-19-190618.txt')
