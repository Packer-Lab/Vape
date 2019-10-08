import numpy as np
import matplotlib.pyplot as plt
import pickle
from utils_funcs import d_prime
import seaborn as sns
import scipy
import sys
import re
     
class Subsets():
    def __init__(self, run):

        self.trial_info = run.trial_info
        self.outcome = run.outcome
        self.get_outcomes()

        if run.session.task_name == 'opto_stim_scope_threeway':
            self.trial_subsets = self.get_trial_subsets_easytest()
        else:
            self.trial_subsets = self.get_trial_subsets()

        self.subset_sizes = np.unique(self.trial_subsets)

    def get_trial_subsets(self):
        
        trial_subsets = []

        for i, info in enumerate(self.trial_info):
            if 'Nogo Trial' not in info:
                try:
                    trial_subset = float(info.split(' ')[info.split(' ').index('stimulating') + 1])
                except:
                    raise ValueError('This is likely not a subset cells experiment')

                trial_subsets.append(trial_subset)

        return np.array(trial_subsets)
        
    def get_trial_subsets_easytest(self):

        trial_subsets = []
        subsets_paths = []

        for i, info in enumerate(self.trial_info):
            if 'Nogo Trial' in info:
                trial_subsets.append(0)
                subsets_paths.append('')
            elif 'all_cells_stimulated' in info:
                trial_subsets.append(150)
                subsets_paths.append('')
            elif 'Subset cells experiment' in info:
                trial_subset = int(re.search('(?<=stimulating )(.*)(?= cells)', info).group(0))
                trial_subsets.append(trial_subset)
                subsets_path = info.split('File path is ')[-1]

        return np.array(trial_subsets)


    def get_full_list(self):

        '''temporary function '''

        trial_subsets = []

        for i, info in enumerate(self.trial_info):
            if 'Nogo Trial' not in info:
                try:
                    trial_subset = float(info.split(' ')[info.split(' ').index('stimulating') + 1])
                except:
                    raise ValueError('This is likely not a subset cells experiment')
                trial_subsets.append(trial_subset)
            else:
                trial_subsets.append('Nogo')

        
        return np.array(trial_subsets)


    # @property
    # def go_outcome(self):
            
        # go_outcome = []
        # for t in self.outcome:
            # if t == 'hit':
                # go_outcome.append(True)
            # elif t == 'miss':
                # go_outcome.append(False)
                    
        # return np.array(go_outcome)
       
    # @property
    # def nogo_outcome(self):

        # nogo_outcome = []
            
        # for t in self.outcome:
            # if t == 'fp':
                # nogo_outcome.append(True)
            # elif t == 'cr':
                # nogo_outcome.append(False)

    def get_outcomes(self):
        self.go_outcome = []
        self.nogo_outcome = []

        for t in self.outcome:
            if t == 'hit':
                self.go_outcome.append(True)
            elif t == 'miss':
                self.go_outcome.append(False)
            elif t =='cr':
                self.nogo_outcome.append(False)
            elif t == 'fp':
                self.nogo_outcome.append(True)

        self.go_outcome = np.array(self.go_outcome)
        self.nogo_outcome = np.array(self.nogo_outcome)


    @property
    def subsets_dprime(self):
        
           
        # assert len([t for t in self.trial_type if t== 'go']) == len(self.trial_subsets) == len(self.go_outcome)
        
        subset_outcome = []
        fp_rate = sum(self.nogo_outcome) / len(self.nogo_outcome)

        for sub in self.subsets:
            subset_idx = np.where(self.trial_subsets == sub)[0]
            print(len(subset_idx))
            subset_outcome.append(sum(self.go_outcome[subset_idx]) / len(subset_idx))
            
        subsets_dprime = [d_prime(outcome, fp_rate) for outcome in subset_outcome]

        return subsets_dprime 


    # @property
    # def running_fa(self):
    
        # outcome = self.outcome
        # # need to look at this
        # nogo_outcome = []

        # for trial in outcome:
            # if trial == 'fp':
                # nogo_outcome.append(1)
            # elif trial == 'cr':
                # nogo_outcome.append(0)
        
        # running_sum = 0
        # running_fa = []
        # for i,trial in enumerate(nogo_outcome):
           # running_sum += trial
           # running_fa.append(running_sum/(i+1))

        # return running_fa

    
    @property
    def subset_running_hit(self):

        outcome = self.outcome
        go_outcome = []
        for trial in outcome:
            if trial == 'hit':
                go_outcome.append(1)
            elif trial == 'miss':
                go_outcome.append(0)

        go_outcome = np.array(go_outcome)
        
 
        subset_running_hit = [] 

        for sub in self.subsets:
            sub_idx = np.where(self.trial_subsets==sub)[0]
            
            running_sum = 0
            running_hit = []
            for i,trial in enumerate(go_outcome[sub_idx]):
               running_sum += trial
               running_hit.append(running_sum/(i+1))

            subset_running_hit.append(running_hit)

        return subset_running_hit 

    def snip_session(self, start=0, end=None):
        '''only take a subset of trials into property calculations above'''
        self.go_outcome = self.go_outcome[start:end]
        self.nogo_outcome = self.nogo_outcome[start:end]
        self.trial_subsets = self.trial_subsets[start:end]

        
       
def analyse_subsets(mouse_id, run_numbers):
    '''gets the value for the variable variable_name on each of the run numbers''' 
    object_list = []
    for run_number in run_numbers:

        with open('/home/jamesrowland/Documents/Code/Vape/run_pkls/{}/run{}.pkl'\
        .format(mouse_id,run_number), 'rb') as f:

            run = pickle.load(f)

        subsets_obj = Subsets(run)
        object_list.append(subsets_obj)
           
    # temporarily returning the varaible run (not in :wst) for some quick analysis
    # in the future, you want each run pkl to be availble easily to notebooks
    return object_list[0], run


def subset_attr(attr_name, object_list):
    attr_list = []
    for obj in object_list:
        attr_list.append(getattr(obj, attr_name))

    return attr_list
        
    
