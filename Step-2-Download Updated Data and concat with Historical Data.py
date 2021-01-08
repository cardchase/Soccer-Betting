import pandas as pd
from datetime import datetime
import numpy as np

start = datetime.now()

print("Downloading data for Season 2020")
Season_2020 = pd.read_excel(r"https://football-data.co.uk/mmz4281/2021/all-euro-data-2020-2021.xlsx", engine='openpyxl',sheet_name=None)
print("Downloading data for Season 2020 completed successfully")
print("Concatenating data for Season 2020")
Season_2020_c = pd.concat(Season_2020, axis=0, ignore_index=True)
print("Concatenating data for Season 2020 completed successfully")
print("Using saved database till last season")
Season93_20 = pd.read_csv(r'path', low_memory=False)
print("Loading saved database complete")
print("Filling missing team names with HT and AT columns")
Season93_20.HomeTeam = Season93_20['HomeTeam'].fillna(Season93_20['HT'])
Season93_20.AwayTeam = Season93_20['AwayTeam'].fillna(Season93_20['AT'])
print("Dropping excess columns")
Season93_20 = Season93_20.drop(
    ['HT', 'AT', 'Unnamed: 52', 'Unnamed: 53', 'Unnamed: 49', 'Unnamed: 50', 'Unnamed: 51', 'Unnamed: 54',
     'Unnamed: 31', 'Unnamed: 32', 'Unnamed: 33', 'Unnamed: 44', 'Unnamed: 45', 'Unnamed: 46', 'Unnamed: 30',
     'Unnamed: 47', 'Unnamed: 48', 'Unnamed: 28', 'Unnamed: 29', 'Unnamed: 25', 'Unnamed: 26', 'Unnamed: 27',
     'Unnamed: 22', 'Unnamed: 23', 'Unnamed: 24', 'Unnamed: 19', 'Unnamed: 20', 'Unnamed: 21'
     ], axis=1)
print("Loading historical data complete")
print("Defining list of all seasons")
Season_list = [Season_2020_c, Season93_20]
print("Concatenating all data into one dataframe")
db_concat = pd.concat(Season_list)
print("Filtering out null date rows")
db_concat = db_concat.dropna(subset=['Date'])
print("Defining Column Types")
column_dict = {'HTR': str,
               'Referee': str,
               'BbAH': int,
               'BbAHh': float,
               }
db_concat = db_concat.astype(column_dict, errors='ignore')
print("Dropping duplicates and cleaning dataset")
db_concat = db_concat.drop_duplicates()
print("Saving csv")
db_concat.to_csv(r"C:\Users\Harshad\Documents\Project\Files\dataset_updated.csv")
print("File Saved")
print("Success")
end = datetime.now()
time_taken = end - start
print('Time taken to complete: ', time_taken)
