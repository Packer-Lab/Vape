import pickle
import os.path
import sys
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pandas as pd
import sys

CRED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'credentials.json')
TOKEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'token.pickle')

def build_gsheet(SPREADSHEET_ID, SHEET_NAME):

    ''' Takes input of google sheets SPREADSHEET_ID and SHEET_NAME
    returns gsheet object that can be read by gsheet2df into a pandas dataframe. 
    This object can also be accessed directly.
    This function is a slightly modified version of quickstart.py 
    (https://developers.google.com/sheets/api/quickstart/python)
    The user must follow Step 1 in this link to enable the google sheets API 
    in their account and download credentials.json  to their working directory.
    JR

    '''
    creds = None
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.

    if os.path.exists(TOKEN_PATH):
        with open(TOKEN_PATH, 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CRED_PATH,
                                                             SCOPES)
            # JR - this is needed to authenticate through ssh
            creds = flow.run_console()
        # Save the credentials for the next run
        with open(TOKEN_PATH, 'wb') as token:
            pickle.dump(creds, token)

    service = build('sheets', 'v4', credentials=creds, cache_discovery=False)

    # Call the Sheets API
    sheet = service.spreadsheets()
    gsheet = sheet.values().get(spreadsheetId=SPREADSHEET_ID,
                                range=SHEET_NAME).execute()

    return gsheet


def gsheet2df(SPREADSHEET_ID, HEADER_ROW, SHEET_NAME='Sheet1!A:F'):
    '''
    Imports the sheet defined in SPREADSHEET_ID as a pandas dataframe
    Inputs -
    SPREADSHEET_ID: found in the spreadsheet URL 
                    https://docs.google.com/spreadsheets/d/SPREADSHEET_ID
    HEADER_ROW:     the row that contains the header - titles of the columns
    SHEET_NAME:     the name of the sheet to import (defaults to Sheet1 the 
                    default sheet name in gdocs)
    Returns -
    df: a pandas dataframe

    Converts gsheet object from build_gsheet to a Pandas DataFrame.
    Use of this function requires the user to follow the instructions 
    in build_gsheet. This function is adapted from 
    https://towardsdatascience.com/how-to-access-google-sheet-data-using-the-
    python-api-and-convert-to-pandas-dataframe-5ec020564f0e
    empty cells are represented by ''

    '''

    gsheet = build_gsheet(SPREADSHEET_ID, SHEET_NAME)

    header = gsheet.get('values', [])[HEADER_ROW-1]

    values = gsheet.get('values', [])[HEADER_ROW:]

    if not values:
        print('no data found')
        return

    # corrects for rows which end with blank cells
    for i, row in enumerate(values):
        if len(row) < len(header):
            [row.append('') for i in range(len(header)-len(row))]
            values[i] = row

    all_data = []
    for col_id, col_name in enumerate(header):

        column_data = []

        for row in values:
            column_data.append(row[col_id])

        ds = pd.Series(data=column_data, name=col_name)
        all_data.append(ds)

    df = pd.concat(all_data, axis=1)

    return df


def correct_behaviour_df(df, t_series_header='t-series name'):
    '''
    inputs the Optimstim Behaviour Metadata
    corrects blank rows that have been merged in gsheets and converts
    lists of t series names from newlines to python lists

    '''
    # fix blank merged rows
    for row, val in enumerate(df['Date']):

        if val:
            continue

        if df.loc[row]['Run Number']:
            for col in df.columns.values[0:4]:
                df.loc[row, col] = df.loc[row-1, col]

        else:
            df = df.drop(row, axis=0)

    # fix newline lists
    for column_header in [t_series_header, 't-series name']:
        for row, val in enumerate(df[column_header]):

            if '\n' in val:
                t_list = val.split('\n')

                # get rid of blank strings
                t_list = [t for t in t_list if t]

                df[column_header][row] = t_list

    return df


def path_finder(umbrella, *args,  is_folder=False):
    '''
    returns the path to the single item in the umbrella folder
    containing the string names in each arg
    is_folder = False if args is list of files
    is_folder = True if  args is list of folders
    '''
    # list of bools, has the function found each argument?
    # ensures two folders / files are not found
    found = [False] * len(args)
    # the paths to the args
    paths = [None] * len(args)
    
    if is_folder:
        for root, dirs, files in os.walk(umbrella):
            for folder in dirs:
                for i, arg in enumerate(args):
                    if arg in folder:
                        assert not found[i], 'found at least two paths for {},'\
                                             'search {} to find conflicts'\
                                             .format(arg, umbrella)
                        paths[i] = os.path.join(root, folder)
                        found[i] = True

    elif not is_folder:
        for root, dirs, files in os.walk(umbrella):
            for file in files:
                for i, arg in enumerate(args):
                    if arg in file:
                        assert not found[i], 'found at least two paths for {},'\
                                             'search {} to find conflicts'\
                                             .format(arg, umbrella)
                        paths[i] = os.path.join(root, file)
                        found[i] = True
    
    print(paths)
    for i, arg in enumerate(args):
        if not found[i]:
            raise ValueError('could not find path to {}'.format(arg))

    return paths


#### slicing / indexing functions #####
def split_df(df, col_id):
    '''slice whole dataframe by boolean value of col_id'''
    idx = df.index[df[col_id] == 'TRUE']
    return df.loc[idx, :]


def df_bool(df, col_id):
    '''returns the rows of a pandas dataframe where col_id is TRUE'''
    return df.index[df[col_id] == 'TRUE']


def df_col(df, col, idx='all'):
    '''returns the values in col in rows specified in idx'''

    if idx == 'all':
        idx = range(len(df))

    try:
        return [df.loc[idx, col] for idx in idx]
    except KeyError:
        print('Spreadsheet does not have column "{}"'.format(col))
        return None


def path_conversion(path_list, packerstation_path):
    '''converts local paths on 2p imaging computer to packerstation paths
       only works with data arrangement as of 2019-03-27 will likely break
       in the future indeed, broken for vastly different paths (user, date,
       base folder etc.)In the same path_list 2019-05-24 (RL)
       to fix: find the 'Data' string and if it is within the first 
       p.split('\\') then it is Packer1 style,
       if in second p.split('\\') it is in PackerStation style
    '''

    converted_paths = []

    for p in path_list:
        if not p:
            raise Exception('ERROR: Path missing')
        elif p.split('\\')[1] == 'Data':
            # Packer1 path
            name = (p.split('\\')[2])
        elif p.split('\\')[2] == 'Data':
            # PackerStation path
            name = (p.split('\\')[1])
        else:
            # Could not recognise path style
            raise Exception(
                'ERROR: Could not recognise path style, make it Packer1 '\
                'or PackerStation friendly')

        date = p.split('\\')[3]
        local_path = os.path.join(packerstation_path, name, 'Data', date)
        converted_path = os.path.join(local_path, *p.split('\\')[4:])
        converted_path = converted_path.replace(
            '"', '')  # get rid of weird quote marks
        converted_paths.append(converted_path)

    return converted_paths
