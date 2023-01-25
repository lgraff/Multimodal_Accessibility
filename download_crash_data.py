# libraries
import util_functions as ut
import pandas as pd
import gc
import os

# get last 2 years of crash data
def download_crash_data(site, resource_ids, output_path):
    # two years of crash data (function can [should] be generalized to include more years) 
    crash_data_0 = ut.get_resource_data(site,resource_id=resource_ids[0],count=999999999) 
    crash_data_1 = ut.get_resource_data(site,resource_id=resource_ids[1],count=999999999) 

    # Convert to pandas df and concatenate
    df_crash_0 = pd.DataFrame(crash_data_0)
    df_crash_1 = pd.DataFrame(crash_data_1)
    df_crash = pd.concat([df_crash_1, df_crash_0], ignore_index=True)
    del crash_data_0  # deleting original data b/c large
    del crash_data_1
    del df_crash_0  # deleting year by year data b/c large
    del df_crash_1
    gc.collect()
    cols_keep = ['DEC_LAT', 'DEC_LONG', 'BICYCLE', 'BICYCLE_COUNT', 'PEDESTRIAN', 'PED_COUNT', 
                'SPEED_LIMIT', 'VEHICLE_COUNT', 'TOT_INJ_COUNT']
    df_crash = df_crash[cols_keep]

    # save as .csv file
    df_crash.to_csv(output_path)

 #   df_crash_2020.to_csv(os.path.join(cwd,'Input_Data','df_crash_2020.csv'))
 #   df_crash_2019.to_csv(os.path.join(cwd,'Input_Data','df_crash_2019.csv'))

# site = "https://data.wprdc.org"
# resource_ids = ["514ae074-f42e-4bfb-8869-8d8c461dd824","cb0a4d8b-2893-4d20-ad1c-47d5fdb7e8d5"]
# download_crash_data(site, resource_ids)