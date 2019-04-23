import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pandas as pd


def build_gsheet(SPREADSHEET_ID, SHEET_NAME):

    """
    Takes input of google sheets SPREADSHEET_ID and SHEET_NAME
    returns gsheet object that can be read by gsheet2df into a pandas dataframe. This object can also be accessed directly.

    This function is a slightly modified version of quickstart.py (https://developers.google.com/sheets/api/quickstart/python)
    The user must follow Step 1 in this link to enable the google sheets API in their account and download credentials.json
    to their working directory.
    JR

    """
    creds = None
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server()
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('sheets', 'v4', credentials=creds)

    # Call the Sheets API
    sheet = service.spreadsheets()
    gsheet = sheet.values().get(spreadsheetId=SPREADSHEET_ID,
                                range=SHEET_NAME).execute()

    return gsheet


def gsheet2df(SPREADSHEET_ID, HEADER_ROW, SHEET_NAME='Sheet1'):

    '''
    Imports the sheet defined in SPREADSHEET_ID as a pandas dataframe
    Inputs -
    SPREADSHEET_ID: found in the spreadsheet URL https://docs.google.com/spreadsheets/d/SPREADSHEET_ID
    HEADER_ROW:     the row that contains the header - titles of the columns
    SHEET_NAME:     the name of the sheet to import (defaults to Sheet1 the default sheet name in gdocs)

    Returns -
    df: a pandas dataframe

    Converts gsheet object from build_gsheet to a Pandas DataFrame.
    Use of this function requires the user to follow the instructions in build_gsheet
    This function is adapted from https://towardsdatascience.com/how-to-access-google-sheet-data-using-the-python-api-and-convert-to-pandas-dataframe-5ec020564f0e

    empty cells are represented by ''

    '''

    gsheet = build_gsheet(SPREADSHEET_ID, SHEET_NAME)

    header = gsheet.get('values', [])[HEADER_ROW-1]

    values = gsheet.get('values', [])[HEADER_ROW:]

    if not values:
        print('no data found')
        return

    #corrects for rows which end with blank cells
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
    #fix blank merged rows
    for row,val in enumerate(df['Date']):

        if val: continue

        for col in df.columns.values[0:4]:

            df.loc[row,col] = df.loc[row-1,col]


    # fix newline lists
    for column_header in [t_series_header, 'Number of frames']:
        for row, val in enumerate(df[column_header]):

            if '\n' in val:
                t_list = val.split('\n')

                #get rid of blank strings
                t_list = [t for t in t_list if t]

                df[column_header][row] = t_list


    return df
