import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import csv
import sys
sys.path.append('..')
import data_import as di
from utils_funcs import *
from gsheets_importer import gsheet2df, correct_behaviour_df

class OptoStimImport1p():

    #ID of the Optistim Behaviour gsheet (from the URL)
    sheet_ID = '1GG5Y0yCEw_h5dMfHBnBRTh5NXgF6NqC_VVGfSpCHroc'

    def __init__(self, ID):

        '''
        class to parse data from 1-photon pycontrol task
        DataFrame
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







if __name__ == "__main__":
    importer = OptoStimImport('RL019')
