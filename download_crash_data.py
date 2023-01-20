# libraries
import util_functions as ut
import pandas as pd
import os

# get last 2 years of crash data
def download_crash_data(site, resource_ids):
    # two years of crash data
    # function can be modified to include more years 
    crash_data_2020 = ut.get_resource_data(site,resource_id=resource_ids[0],count=999999999) 
    crash_data_2019 = ut.get_resource_data(site,resource_id=resource_ids[1],count=999999999) 

    # Convert to pandas df and concatenate
    df_crash_2020 = pd.DataFrame(crash_data_2020)
    df_crash_2019 = pd.DataFrame(crash_data_2019)

    del crash_data_2020  # deleting original data b/c large
    del crash_data_2019

    # save as .csv files in Input Data folder
    cwd = os.getcwd()
    df_crash_2020.to_csv(os.path.join(cwd,'Input_Data','df_crash_2020.csv'))
    df_crash_2019.to_csv(os.path.join(cwd,'Input_Data','df_crash_2019.csv'))

# site = "https://data.wprdc.org"
# resource_ids = ["514ae074-f42e-4bfb-8869-8d8c461dd824","cb0a4d8b-2893-4d20-ad1c-47d5fdb7e8d5"]
# download_crash_data(site, resource_ids)