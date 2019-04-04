import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import csv
import utils.data_import as di

class opto_stim_import():
    def __init__(self, ID, path, date, task_str = None):
        '''
        class to parse relevant opto_stim task information to a python object
        inputs:
        ID - the ID of the mouse 
        path - path to the behavioural txt files, must point to directory of date folders containing txt files 
        date - the date of interest must be in same format as folders in path
        task_str - allows user to specify a string if there is more than 1 txt file for a single mouse on 1 day
        
        '''
        self.ID = ID
        self.path = path
        self.date = date
        self.date_path = os.path.join(path,date)
        self.task_str = task_str
        
        self.get_files()
        
        #use data_import to build session do not add this to object 
        self.session = di.Session(self.txt_file)

        self.print_lines = self.session.print_lines
        
        self.trial_outcome()
        self.gonogo()
        self.num_trials = len(self.outcome)
        print('{0} trials total'.format(self.num_trials))
        
        self.start_times()

        
        self.get_current()

        self.get_licks()
        
        self.dprime()
        
        self.autoreward()
        
        self.manual_correct()
        

        
        
    def get_files(self):
        '''get text files for ID for given date in path'''
        os.chdir(self.date_path)
        txt_files = [file for file in glob.glob("*.txt") if self.ID in file]
        # filter txt files if task_str given
        if self.task_str:
            txt_files = [file for file in txt_files if self.task_str in file]
        
        
        
        #make sure haven't returned more than 1 txt file
        assert len(txt_files) == 1, 'error finding txt files'
 
        self.txt_file = txt_files[0]
        self.txt_file = txt_files[0]

    def trial_outcome(self):
        '''creates list of strings of trial outcomes and the times they occured'''
        outcome = []
        trial_time = []
        for line in self.print_lines:
            if 'earned_reward' in line:
                time = float(line.split(' ')[0])
                trial_time.append(time)
                outcome.append('hit')
            elif 'missed trial' in line:
                time = float(line.split(' ')[0])
                trial_time.append(time)
                outcome.append('miss')            
            elif 'correct rejection' in line:
                time = float(line.split(' ')[0])
                trial_time.append(time)
                outcome.append('cr')
            elif 'false positive' in line:
                time = float(line.split(' ')[0])
                trial_time.append(time)
                outcome.append('fa')
                
        self.trial_time, self.outcome = self.debounce(trial_time, outcome, 2000)
        self.trial_time = np.array(self.trial_time)
        
    def gonogo(self):
        '''generates a list of trial type (go or nogo) based on outcome'''
        self.trial_type = ['go' if t == 'miss' or t == 'hit' else 'nogo' for t in self.outcome]
        
    def debounce(self, arr1, arr2, window):
        '''
        function to debounce arrays based on the numeric values of arr1, used for example when the trial outcome is erroneously printed twice
        removes subsequent elements of an array that occur within a time window to the previous element
        '''
        for i,a in enumerate(arr1):
            # finish debouncing if reach the final element
            if i == len(arr1) - 1:
                return arr1, arr2

            else:
                diff = arr1[i+1] - arr1[i]
                if diff <= window:
                    del arr1[i+1]
                    del arr2[i+1]
                    # recursively call debounce function
                    self.debounce(arr1,arr2, window)
        
        
    
    
    def get_current(self):
        '''gets the LED current on each trial'''
        
        self.LED_current = [float(line.split(' ')[4]) for line in self.print_lines if 'LED current is' in line and 'now' not in line]
        
        if self.session.task_name == 'opto_stim_psychometric':
            self.LED_current = [float(line.split(' ')[3]) for line in self.print_lines if 'LED_current is' in line]
            
        
        #assert len(self.LED_current) == self.num_trials, 'Error, num LED currents is {} and num trials is {}'.format(len(self.LED_current), self.num_trials)

    def start_times(self):
        '''the time that the trial started'''
        self.trial_start =  [float(line.split(' ')[0]) for line in self.print_lines if 'goTrial' in line or 'nogo_trial' in line]
    
    def dprime(self):
        '''get the value of online dprime calculated in the task'''
        self.online_dprime = [float(line.split(' ')[3]) for line in self.print_lines if 'd_prime is' in line]
       
    
    def get_licks(self):
        '''gets the lick times normalised to the start of each trial'''
        licks = self.session.times.get('lick_1')
        
        self.binned_licks = []
        
        for i,t in enumerate(self.trial_start):
            t_start = t
            if i == len(self.trial_start)-1:
                # arbitrary big number to prevent index error on last trial
                t_end = 10e100
            else:
                t_end = self.trial_start[i+1]
                
            #find the licks occuring in each trial    
            trial_IND = np.where((licks>t_start) & (licks<t_end))[0]
            
            #normalise to time of trial start
            trial_licks = licks[trial_IND] - t_start

            self.binned_licks.append(trial_licks)

            
    def autoreward(self):
        '''detect if pycontrol went into autoreward state '''
        autoreward_times = self.session.times.get('auto_reward') 
        self.autorewarded_trial = []
        for i,t in enumerate(self.trial_time):
            t_start = t
            if i == self.num_trials-1:
                # arbitrary big number to prevent index error on last trial
                t_end = 10e100
            else:
                t_end = self.trial_time[i+1]


            is_autoreward = [a for a in autoreward_times if a >= t_start and a < t_end]
            if is_autoreward:
                self.autorewarded_trial.append(True)
            else:
                self.autorewarded_trial.append(False)
                
        ## did mouse get out of autreward phase?
        time_autoswitch = False
        self.trial_autoswitch = False
        for line in self.print_lines:
            if 'switching out of auto' in line:
                time_autoswitch = float(line.split(' ')[0])
                break
     
                
        ## find the trial number this happened by finding the index of the closest trial start time (+1)
        if time_autoswitch:
            self.trial_autoswitch = np.abs(self.trial_time - time_autoswitch).argmin() + 1
            
    def manual_correct(self):
        '''to be deprecated function to manually correct early versions of task where some variables were not printed'''
        if self.task_str == 'CBCB1213.1b-2018-10-15-194755':
            self.LED_current = [53.73] * self.num_trials
        elif self.task_str == 'CBCB1213.1b-2018-10-16-120105':
            # did not print the LED current, need to add manually based on number of steps down
            trial_LED_change = []
            current_list = [53.73, 37.91, 27.36, 14.17, 11.53, 6.26, 4.94, 3.62, 2.3, 2.04, 1.78, 1.51]
            for line in self.print_lines:
                if 'Changing LED current' in line:
                    time_LED_change = float(line.split(' ')[0])
                    trial_LED_change.append(np.abs(self.trial_time - time_LED_change).argmin())
                    
            trial_LED_change.insert(0,0)
            # the number of trials a mouse was at a given power level
            t_diff = np.diff(np.array(trial_LED_change))
            
            for i,t in enumerate(t_diff):
                [self.LED_current.append(current_list[i]) for x in range(t)]
                
            #the end state current for this day
            num_end = self.num_trials - trial_LED_change[-1]
            [self.LED_current.append(current_list[len(trial_LED_change)+1]) for x in range(num_end)]

                
                
class merge_sessions():
    def __init__(self, ID, behaviour_path, LUT_path):
        '''
        class to merge all sessions from a mouse into single variables
        inputs: ID - the mouse of interest
                behaviour_path: path to directory of dates with behaviours
                LUT_path: path to the LUT detailing the names and order of behaviour txts for a mouse
        '''     
        
        self.ID = ID
        self.behaviour_path = behaviour_path
        self.LUT_path = LUT_path
        
        self.build_path_dict()
        
        self.merge()   

        self.total_trials = len(self.outcome)
        print('Across sessions, mouse {0} has done {1} trials'.format(self.ID, self.total_trials))
        
        #assert len(self.outcome) == len(self.online_dprime) == len(self.binned_licks) == len(self.LED_current) == len(self.autorewarded_trial)

        
    def build_path_dict(self):
        #a dictionary containing the keys of mouse names and the paths to the txt files as vals
        path_dict = {}

        with open(self.LUT_path, 'r') as csvfile:
            LUTreader = csv.reader(csvfile, delimiter=',')
            for row in LUTreader:
                path_dict[row[0]] = row[1:]
                
        # the txt files for the mouse of interest
        try:
            self.mouse_txts = path_dict[self.ID]
        except:
            raise ValueError('mouse not present in behavioural LUT')
        
    def merge(self):  
        '''merge task info from seperate session into a single list probably a neater way of doing this'''
        self.outcome = []
        self.online_dprime = []
        self.binned_licks = [] 
        self.LED_current = []
        self.autorewarded_trial = []
        self.trial_type = []
        self.trial_start = []
        #count the total number of trials across session before mouse gets out of autoreward 
        self.trial_autoswitch = 0
        #switch on/off counting sessions once mouse has switched out once
        count_auto = True
        
        for txt in self.mouse_txts:
            #throw out blank cells
            if txt:
                date = self.extract_date(txt)
                # future refinement could super or sub this class
                t_session = opto_stim_import(self.ID, self.behaviour_path, date, task_str = txt)
                [self.outcome.append(x) for x in t_session.outcome]
                [self.online_dprime.append(x) for x in t_session.online_dprime]
                [self.binned_licks.append(x) for x in t_session.binned_licks]
                [self.LED_current.append(x) for x in t_session.LED_current]
                [self.autorewarded_trial.append(x) for x in t_session.autorewarded_trial]
                [self.trial_type.append(x) for x in t_session.trial_type]
                [self.trial_start.append(x) for x in t_session.trial_start]
                # if the mouse didn't get out of autoreward, add the total number of trials
                # if it did, add the number of trials it took then stop counting
                
                if count_auto and not t_session.trial_autoswitch:
                    self.trial_autoswitch += t_session.num_trials
                elif count_auto and t_session.trial_autoswitch:
                    self.trial_autoswitch += t_session.trial_autoswitch
                    count_auto = False


    def extract_date(self, txt):
        '''clunky function to extract date directly from txt file name this may break at some point so check'''
        txt_split = txt.split('-')

        return '{0}-{1}-{2}'.format(txt_split[1], txt_split[2], txt_split[3])

    
    
