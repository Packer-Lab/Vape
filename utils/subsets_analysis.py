import numpy as np
import matplotlib.pyplot as plt
import pickle
from utils.utils_funcs import d_prime
import seaborn as sns
import scipy
import sys
sys.path.append('..')
     
class Subsets():
    def __init__(self, run):

        self.run = run
        self.subsets = np.unique(self.trial_subsets)

    @property
    def trial_subsets(self):
        
        trial_subsets = []

        for i, info in enumerate(self.run.trial_info):
            if 'Nogo Trial' not in info:
                trial_subset = float(info.split(' ')[info.split(' ').index('stimulating') + 1])
                trial_subsets.append(trial_subset)
        
        return np.array(trial_subsets)

    @property
    def go_outcome(self):
            
        go_outcome = []
        for t in self.run.outcome:
            if t == 'hit':
                go_outcome.append(True)
            elif t == 'miss':
                go_outcome.append(False)
                    
        return np.array(go_outcome)
       

    @property
    def subsets_dprime(self):
           
        assert len([t for t in self.run.trial_type if t== 'go']) == len(self.trial_subsets) == len(self.go_outcome)
        
        subset_outcome = []

        for sub in self.subsets:
            subset_idx = np.where(self.trial_subsets == sub)[0]

            subset_outcome.append(sum(self.go_outcome[subset_idx]) / len(subset_idx))
            
        subsets_dprime = [d_prime(outcome, self.run.fp_rate) for outcome in subset_outcome]

        return subsets_dprime 


    @property
    def running_fa(self):
    
        outcome = self.run.outcome
        nogo_outcome = []

        for trial in outcome:
            if trial == 'fp':
                nogo_outcome.append(1)
            elif trial == 'cr':
                nogo_outcome.append(0)
        
        running_sum = 0
        running_fa = []
        for i,trial in enumerate(nogo_outcome):
           running_sum += trial
           running_fa.append(running_sum/(i+1))

        return running_fa

    
    @property
    def subset_running_hit(self):

        outcome = self.run.outcome
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
        
       
def analyse_subsets(mouse_id, run_numbers, variable_name):
    '''gets the value for the variable variable_name on each of the run numbers''' 
    object_list = []
    for run_number in run_numbers:

        with open('/home/jamesrowland/Documents/Code/Vape/run_pkls/{}/run{}.pkl'\
        .format(mouse_id,run_number), 'rb') as f:

            run = pickle.load(f)

        subsets_obj = Subsets(run)
        object_list.append(subsets_obj)
            
    return object_list


def subset_attr(attr_name, object_list):
    attr_list = []
    for obj in object_list:
        attr_list.append(getattr(obj, attr_name))

    return attr_list
        
    
