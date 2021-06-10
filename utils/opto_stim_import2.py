import warnings
import numpy as np
import os
import sys
sys.path.append('..')
from data_import import Session
from utils_funcs import paq_data
from utils_funcs import d_prime as pade_dprime
import gsheets_importer as gsheet
from paq2py import paq_read
from rsync_aligner import Rsync_aligner
import re
from ntpath import basename
from urllib.error import HTTPError
from googleapiclient.errors import HttpError
warnings.filterwarnings("ignore")


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
        return [float(line.split(' ')[0]) for line in 
                self.print_lines if str_ in line]

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

        self.trial_time, self.outcome = OptoStimBasic._debounce(
                                                       self.trial_time,
                                                       self.outcome,
                                                       2000
                                                       )
        self.trial_time = self.test_import_and_slice(self.trial_time)
        self.outcome = self.test_import_and_slice(self.outcome)

    @classmethod
    def _debounce(cls, arr1, arr2, window):
        '''
        function to debounce arrays based on the numeric values of arr1, 
        used for example when the trial outcome is erroneously printed twice
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
        trial_type = ['go' if t == 'miss' or t == 'hit' 
                      else 'nogo' for t in self.outcome]
        return self.test_import_and_slice(trial_type)

    @property
    def online_dprime(self):
        return [float(line.split(' ')[3]) for line in self.print_lines 
                if 'd_prime is' in line]

    @property
    def binned_licks(self):
        ''' gets the lick times normalised to the start of each trial '''
        licks = self.session.times.get('lick_1')

        binned_licks = []
        for i, t_start in enumerate(self.trial_start):

            # arbitrary big number to prevent index error on last trial
            if i == len(self.trial_start)-1:
                t_end = 10e100
            else:
                t_end = self.trial_start[i+1]

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
        test that import has been done correctly and slice to remove end 
        trial if IndexError in task tests that list_if of correct length
        and slices to only include full trial_start
        args = list to test for correct length and slice
        '''

        len_test = lambda x : True if self.n_trials_complete\
                              <= x <= self.n_trials_complete + 1 else False

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
                LED_current = [float(line.split(' ')[4]) for line in self.print_lines 
                               if 'LED current is' in line and 'now' not in line]
            elif self.session.task_name == 'opto_stim_psychometric':
                LED_current =  [float(line.split(' ')[3]) for line in self.print_lines 
                                if 'LED_current is' in line]
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

                is_autoreward = next((a for a in autoreward_times if 
                                      a >= t_start and a < t_end), False)
                if is_autoreward: self.autorewarded_trial.append(True)
                else: self.autorewarded_trial.append(False)

            ## did mouse get out of autreward phase?
            
            time_autoswitch = self.pl_times('switching out of auto')
                
            ## find the trial number this happened by finding the index 
            ## of the closest trial start time (+1)
            if time_autoswitch:
                self.trial_autoswitch = np.abs(self.trial_time -
                                               time_autoswitch[0]).argmin() + 1
            else:
                self.trial_autoswitch = None

        @property
        def trial_complete(self):
            time_complete =  self.pl_times('mouse has completed task to lowest power level')

            if time_complete:
                return np.abs(self.trial_time - time_complete[0]).argmin() + 1
            else:
                return None


        @property
        def trial_start(self):
            # go_start = self.session.times.get('detect_lick_go')
            # nogo_start = self.session.times.get('detect_lick_nogo')

            go_start = [int(line.split(' ')[0]) for line in self.print_lines
                        if 'goTrial' in line]

            nogo_start = [int(line.split(' ')[0]) for line in self.print_lines
                          if 'nogo_trial' in line]

            ts = np.sort(np.hstack((go_start, nogo_start)))

            return self.test_import_and_slice(ts) 
            


class OptoStim2p(OptoStimBasic):
    def __init__(self, txt_path):
        '''init this class to process the 2p opto_stim txt file in txt_path'''

        super().__init__(txt_path)

        #the line that triggers an SLM trial throuh blimp
        _slm_trigger_lines = [line for line in self.print_lines 
                              if 'Trigger SLM trial' in line]
        _nogo_trigger_lines = [line for line in self.print_lines 
                               if 'Trigger NOGO trial' in line]
        _alltrials_trigger_lines = [line for line in self.print_lines 
                                    if 'Trigger SLM trial' in line or 
                                    'Trigger NOGO trial' in line]
                                    
         
        self.slm_barcode = [re.search('(?<=Barcode )(.*)', line).group(0)
                            for line in _slm_trigger_lines]
        self.slm_trial_number = [re.search('(?<=Number )(.*)(?= Barcode)', line)
                                 .group(0) for line in _slm_trigger_lines]

        self.nogo_barcode = [re.search('(?<=Barcode )(.*)', line).group(0) for 
                             line in _nogo_trigger_lines]
        self.nogo_trial_number = [re.search('(?<=Number )(.*)(?= Barcode)', line)
                                  .group(0) for line in _nogo_trigger_lines]

        self.alltrials_barcodes = [re.search('(?<=Barcode )(.*)', line).group(0)
                                   for line in _alltrials_trigger_lines]

        assert len(self.slm_barcode) == self.trial_type.count('go'),\
                  '{} {}'.format(len(self.slm_barcode), self.trial_type.count('go'))
        # necessary to analyse sessions without nogo blimping
        if len(self.nogo_barcode) > 0:
            assert len(self.nogo_barcode) == self.trial_type.count('nogo'),\
                  '{} {}'.format(len(self.nogo_barcode), self.trial_type.count('nogo'))

        self.rsync = self.session.times.get('rsync')
        # go_start = self.session.times.get('SLM_state')
        nogo_start = self.session.times.get('nogo_state')
        easy_start  = self.session.times.get('easy_state')
        test_start  = self.session.times.get('test_state')

        ts = np.sort(np.hstack((easy_start, test_start, nogo_start)))
        self.trial_start = self.test_import_and_slice(ts) 
        # print(self.trial_start)
     

class BlimpImport(OptoStim2p):

    #from the URL
    sheet_IDs = ['1GG5Y0yCEw_h5dMfHBnBRTh5NXgF6NqC_VVGfSpCHroc',
                 '1nFdqJv1aZk36CrBpRZPjRuOTHUKX8SXuVW9pa89OIHY',
                 '1Cnt8-e7rGFMlvkRwkiT2BlWC4Ll6uvC9iO3dVugyMbM']

    server_path = '/home/jrowland/mnt/qnap/Data'

    # the column headers of the spreadsheet 
    date_header = 'Date'
    tseries_header = 't-series name'
    paq_header = '.paq file'
    naparm_header = 'Naparm'
    blimp_header = 'blimp folder'
    pycontrol_header = 'pycontrol txt'
    prereward_header = 'Pre-reward?'
    plane_header = 'Number of planes'
    analyse_bool_header = 'Analyse'

    def __init__(self, mouse_id):

        '''class to import 2-photon and 1-photon blimp experiments based on the
           gsheet given in sheet ID

           '''

        self.mouse_id = mouse_id
        _sheet_name = self.mouse_id + '!A1:ZZ69'

        for idx, sheet_ID in enumerate(BlimpImport.sheet_IDs):
        
            try:
                self.df = gsheet.gsheet2df(sheet_ID, HEADER_ROW=2,
                                               SHEET_NAME=_sheet_name)
                break
            except HttpError:
                if idx != len(BlimpImport.sheet_IDs)-1:
                    continue
                else:
                    print(sheet_ID)
                    raise FileNotFoundError('gsheet id incorrect')

        self.df = gsheet.correct_behaviour_df(self.df)
        self.parse_spreadsheet()

    def parse_spreadsheet(self):
    
        '''parses the spreadsheet for dealing with 2p behaviour
           use import_1p for 1-photon behaviour
            
           '''

        idx_analyse = gsheet.df_bool(self.df, BlimpImport.analyse_bool_header)
        idx_2p = gsheet.df_bool(self.df, 'Trained 2P')
        idx_1p = gsheet.df_bool(self.df, 'Trained 1P')

        intersect = lambda A, B: set(A) - (set(A) - set(B))

        self.rows_2p = intersect(idx_analyse, idx_2p)
        self.rows_1p = intersect(idx_analyse, idx_1p)

        self.dates_2p = gsheet.df_col(self.df, BlimpImport.date_header, 
                                      self.rows_2p)
        self.paqs = gsheet.df_col(self.df, BlimpImport.paq_header,
                                  self.rows_2p)
        self.naparm_folders = gsheet.df_col(self.df, BlimpImport.naparm_header,
                                            self.rows_2p)
        self.blimp_folders = gsheet.df_col(self.df, BlimpImport.blimp_header,
                                           self.rows_2p)
        self.pycontrol_folders = gsheet.df_col(self.df,
                                               BlimpImport.pycontrol_header,
                                               self.rows_2p)
        self.prereward_folders = gsheet.df_col(self.df, BlimpImport.prereward_header, 
                                               self.rows_2p)
        self.tseries_folders = gsheet.df_col(self.df, BlimpImport.tseries_header, 
                                             self.rows_2p)
        self.plane_numbers = gsheet.df_col(self.df, BlimpImport.plane_header, 
                                           self.rows_2p)

        assert len(self.paqs) == len(self.naparm_folders) == \
               len(self.blimp_folders) == len(self.dates_2p)
        num_runs = len(self.paqs)

    def import_1p(self):

        '''call me instead of get_object_and_test if 
           importing 1-photon experiment''' 
        
        dates_1p = gsheet.df_col(self.df, BlimpImport.date_header, self.rows_1p)
        pycontrol_1p = gsheet.df_col(self.df, BlimpImport.pycontrol_header,
                                     self.rows_1p)

        obj_list = []
        for date, pyc in zip(dates_1p, pycontrol_1p):
            umbrella = os.path.join(BlimpImport.server_path, '1-photon-behaviour')
            pycontrol_path = gsheet.path_finder(umbrella, pyc, is_folder=False)
            obj = OptoStim1p(pycontrol_path[0])
            obj_list.append(obj)

        return obj_list


    def get_object_and_test(self, run, raise_error=True):

        '''Build object for a sepcific run and test that details
           have been entered correctly

           Adds attributes to the BlimpImport object from the paq file,
           and blimp alignment file also generates an alignment object 
           using rsync that can be used to map paq events
           to pycontrol txt file events
           Inputs:
           run -- run number to process, matching metadata spreadsheet
           raise_error -- whether to raise an error or just print if the paq
                          or blimp alignment file could not be matched to
                          the pycontrol txt file

           '''

        self.run_pycontrol_txt = self.df.loc[self.df['Run Number'] ==\
                                 str(run)]['pycontrol txt'].tolist()[0]

        run_idx = self.pycontrol_folders.index(self.run_pycontrol_txt)

        date = self.dates_2p[run_idx]
        paq = self.paqs[run_idx]
        naparm = self.naparm_folders[run_idx]
        blimp = self.blimp_folders[run_idx]
        pycontrol = self.pycontrol_folders[run_idx]
        prereward = self.prereward_folders[run_idx]
        tseries = self.tseries_folders[run_idx]
        if self.plane_numbers[run_idx]:
            self.num_planes = int(self.plane_numbers[run_idx])
        else:
            self.num_planes = None

        self.run_idx = run_idx 

        umbrella = os.path.join(BlimpImport.server_path, date)
        print(umbrella)
        self.blimp_path = gsheet.path_finder(umbrella, blimp, is_folder=True)[0]
        # Bit of a hack as often naparm is used from previous days and 
        # so is not in the umbrella
        #naparm_umbrella = os.path.join(BlimpImport.server_path, naparm.split('_')[0])
        # Hack changed so does not rely on underscore. Now assumes that the naparm
        # folder starts with iso-standard date to match the Data umbrella folder
        naparm_umbrella = os.path.join(BlimpImport.server_path, naparm[:10])
        self.naparm_path = gsheet.path_finder(naparm_umbrella, naparm, is_folder=True)
        self.pycontrol_path, self.paq_path, self.prereward_path =\
        gsheet.path_finder(umbrella, pycontrol, paq, prereward, is_folder=False)
        
        if tseries == 'None' or not tseries:
            self.tseries_paths = None
        else:
            self.tseries_paths = [gsheet.path_finder(umbrella, t, is_folder=True)
            for t in tseries]

        with open(os.path.join(self.blimp_path, 'blimpAlignment.txt'), 'r') as f:
            file_lines = f.readlines()
            self.align_barcode = [re.search('(?<=Barcode )(.*)(?=. Info)', line)
                                  .group(0) for line in file_lines] 
            self.trial_info = [line.split('Info:')[-1].rstrip('\n').strip()
                               for line in file_lines]

        # build behaviour object from the pycontrol txt file
        super().__init__(self.pycontrol_path)
        # test that the list of barcodes printed in the pycontrol 
        # sequence is contained in the list of barcodes from the 
        # alignment folder
        if not ''.join([str(b) for b in self.alltrials_barcodes]) in ''.join(
                       [str(b) for b in self.align_barcode]):
            error_str = 'pycontrol {} does not match blimp folder {}'.\
                         format(pycontrol, blimp)
            if raise_error: raise ValueError(error_str)
            else: print(error_str)
        else:
            print('pycontrol {} successfully matched to blimp folder {}'
                  .format(pycontrol, blimp))

        # read the paq file and get out useful info
        _paq_obj = paq_read(self.paq_path)

        self.paq_rsync = paq_data(_paq_obj, 'pycontrol_rsync', 
                                 threshold_ttl=True, plot=False)
        self.frame_clock = paq_data(_paq_obj, 'frame_clock', 
                                    threshold_ttl=True, plot=False)
        self.x_galvo_uncaging = paq_data(_paq_obj, 'x_galvo_uncaging', 
                                         threshold_ttl=False, plot=False)
        self.slm2packio = paq_data(_paq_obj, 'slm2packio', threshold_ttl=True,
                                   plot=False)
        self.paq_rate = _paq_obj['rate']

        try:
            self.aligner = Rsync_aligner(pulse_times_A=self.rsync,
                                        pulse_times_B=self.paq_rsync,
                                        units_B=1000/self.paq_rate, 
                                        chunk_size=6, plot=True, 
                                        raise_exception=True)

            self.paq_correct = True
            print('pycontrol {} rsync successfully matched to paq {}'.
                   format(basename(self.prereward_path), 
                          basename(self.paq_path)))
        except Exception as e:
            print(e)
            self.paq_correct = False
            error_str = 'pycontrol rsync does not match paq'
            if raise_error: raise ValueError(error_str)
            else: print(error_str)

        if prereward:
            self.process_prereward(raise_error=raise_error)

        
    def process_prereward(self, raise_error):

        ''' process the prereward txt file '''

        prereward_session = Session(self.prereward_path) 
        self.pre_rsync = prereward_session.times.get('rsync')
        self.pre_licks = prereward_session.times.get('lick_1')
        self.pre_reward = prereward_session.times.get('reward')

        ## workaround for if accidently stop paq after prereward
        # pre_paq_path = '/home/jrowland/mnt/qnap/Data/2019-12-16/'\
                       # '2019-12-14_J064_t003.paq'

        # _pre_paq = paq_read(pre_paq_path)

        # _pre_paq_rsync = paq_data(_pre_paq, 'pycontrol_rsync', 
                                 # threshold_ttl=True, plot=False)

        try:
            self.prereward_aligner = Rsync_aligner(pulse_times_A=self.pre_rsync,
                                                   pulse_times_B=self.paq_rsync,
                                                   units_B=1000/self.paq_rate, 
                                                   chunk_size=6, plot=False, 
                                                   raise_exception=True)
            self.paq_correct = True
            print('prereward {} rsync successfully matched to paq {}'.
                   format(basename(self.prereward_path), basename(self.paq_path)))
        except:
            self.paq_correct = False
            error_str = 'prereward rsync does not match paq'
            if raise_error: raise ValueError(error_str)
            else: print(error_str)
        
        self.both_aligner =  Rsync_aligner(pulse_times_A = 
                                           np.hstack((self.pre_rsync, 
                                                      self.rsync)), 
                                           pulse_times_B = self.paq_rsync,
                                           units_B = 1000/self.paq_rate,
                                           chunk_size=6, plot=False,
                                           raise_exception=True)


