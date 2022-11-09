#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 13:49:00 2022

@author: lindsaygraff
"""

import matplotlib.pyplot as plt
import numpy as np
import partridge as ptg
import pandas as pd
import scipy.stats
import os
from util_functions import *
import config as conf

config_data = conf.config_data

cwd = os.getcwd()
inpath = os.path.join(cwd, 'Data', 'Input_Data', 'GTFS.zip')

# Partridge example: https://gist.github.com/invisiblefunnel/6015e65684325281e65fa9339a78229b

view = {"trips.txt": {}}

# Choose a date
view["trips.txt"]["service_id"] = ptg.read_busiest_date(inpath)[1]
# Build a working feed (all route-dir pairs). Load the feed once here
feed = ptg.load_feed(inpath, view)

all_routes = feed.trips.route_id.unique()    # all routes
all_dirs = feed.trips.direction_id.unique()  # all directions
# Find trips overlapping the time window
all_trips = feed.stop_times[
    (feed.stop_times.arrival_time >= config_data['Time_Intervals']['time_start']*60*60) # time_start = 7am
    & (feed.stop_times.arrival_time <= config_data['Time_Intervals']['time_end']*60*60) # time_end = 9am
    ].trip_id.unique().tolist()

#%%
df_headway_rows = []
for r in all_routes:
    for d in all_dirs: 
        # filter trips_ids by route and direction
        trip_ids = feed.trips[
            (feed.trips.route_id == r)
            & (feed.trips.direction_id == d)
            & (feed.trips.trip_id.isin(all_trips))
            ].trip_id.unique()
        
        # mask = np.isin(all_trips, trip_ids)
        # trip_ids_interval = all_trips[mask]  # trip ids for the route-dir pair in the specified time interval
        
        # Collect the stop_ids for each trip
        if len(trip_ids) > 0:
            trip_stops = []  # list of sets, where each set contains the stops for a specified trip_id
            for _, stimes in feed.stop_times[feed.stop_times.trip_id.isin(trip_ids)].groupby("trip_id"):
                trip_stops.append(set(stimes.stop_id))
            
            # # Find stop_ids shared between all trips (&= is shorthand for set intersection)
            # common_stops = set(trip_stops[0])
            # for stop_ids in trip_stops[1:]:
            #     common_stops &= stop_ids
            
            # assert common_stops
            
            # any stop for any trip in the time interval for the route-dir pair
            all_stops = set([stop for stop_list in trip_stops for stop in stop_list])
            
            #for s in common_stops:
            for s in all_stops:
                # get the departure / arrival times for specified route-dir-stop
                stimes = feed.stop_times[
                    (feed.stop_times.trip_id.isin(trip_ids))
                    & (feed.stop_times.stop_id == s)
                    ].sort_values('arrival_time')
                # time difference between arrivals of consecutive trips for the given route-dir pair
                headway_seconds = stimes.arrival_time.iloc[1:].values - stimes.arrival_time.iloc[:-1].values
                headway_minutes = (headway_seconds / 60.).round(1)  # .astype(int)
                
                if len(stimes) > 1:  # 1) multiple trips in the interval, can calc headway between them
                    headway_median = np.median(headway_minutes)
                else:  # 2) one trip in the interval, assume headway is the full length of the interval
                    headway_median = config_data['Time_Intervals']['time_end']*60 - config_data['Time_Intervals']['time_start']*60
                row = [r, d, s, headway_median]   #if len(trip_ids_interval) > 1 else [
                    #route, direction, s, 
                df_headway_rows.append(row)
            
df_headway = pd.DataFrame(df_headway_rows, columns=['route_id', 'direction_id', 'stop_id', 'headway_min'])
df_headway.to_csv(os.path.join(cwd, 'Data', 'Output_Data', 'PT_headway_NEW.csv'))     

'''
#%% should not be any null values; check what's going on
null_mask = df_headway.headway_min.isnull().values
# # minutes
null_data = df_headway[null_mask] #= config_data['Time_Intervals']['len_period']

#%% troubleshoot

route, direction = 'Y46', 0

df_headway_test_rows = []
# filter trips_ids by route and direction
trip_ids = feed.trips[
    (feed.trips.route_id == route)
    & (feed.trips.direction_id == direction)
    & (feed.trips.trip_id.isin(all_trips))
    ].trip_id.unique()

# mask = np.isin(all_trips, trip_ids)
# trip_ids_interval = all_trips[mask]  # trip ids for the route-dir pair in the specified time interval

# Collect the stop_ids for each trip
if len(trip_ids) > 0:
    trip_stops = []  # list of sets, where each set contains the stops for a specified trip_id
    for _, stimes in feed.stop_times[feed.stop_times.trip_id.isin(trip_ids)].groupby("trip_id"):
        trip_stops.append(set(stimes.stop_id))
    
    # # Find stop_ids shared between all trips (&= is shorthand for set intersection)
    # common_stops = set(trip_stops[0])
    # for stop_ids in trip_stops[1:]:
    #     common_stops &= stop_ids
    
    # assert common_stops
    
    # any stop for any trip in the time interval for the route-dir pair
    all_stops = set([stop for stop_list in trip_stops for stop in stop_list])
    
    #for s in common_stops:
    for s in ['9401']: # all_stops:
        # get the departure / arrival times for specified route-dir-stop
        stimes = feed.stop_times[
            (feed.stop_times.trip_id.isin(trip_ids))
            & (feed.stop_times.stop_id == s)
            ].sort_values('arrival_time')
        # time difference between arrivals of consecutive trips for the given route-dir pair
        headway_seconds = stimes.arrival_time[1:].values - stimes.arrival_time[:-1].values
        headway_minutes = (headway_seconds / 60.).round(1)  # .astype(int)
        
        if len(trip_ids) > 1:  # 1) multiple trips in the interval, can calc headway between them
            headway_median = np.median(headway_minutes)
        else:  # 2) one trip in the interval, assume headway is the full length of the interval
            headway_median = config_data['Time_Intervals']['time_end']*60 - config_data['Time_Intervals']['time_start']*60
        row = [route, direction, s, headway_median]   #if len(trip_ids_interval) > 1 else [
            #route, direction, s, 
        df_headway_test_rows.append(row)

dfy46 = pd.DataFrame(df_headway_test_rows, columns=['route_id', 'direction_id', 'stop_id', 'headway_min'
                                                   ]).sort_values(by='stop_id')
#%%


#%% Compare old and new 
df_new = pd.read_csv(os.path.join(cwd, 'Data', 'Output_Data', 'PT_headway_NEW.csv'))
df_old = pd.read_csv(os.path.join(cwd, 'Data', 'Output_Data', 'PT_headway_OLD.csv'))

#%%
route_new = set(df_new.route_id.unique().tolist())
route_old = set(df_old.route_id.unique().tolist())

print(route_old - route_new)

df_new['route_dir_stop'] = df_new['route_id'] + '_' + df_new['direction_id'].astype(str) + '_' + + df_new['stop_id']
df_old['route_dir_stop'] = df_old['route_id'] + '_' + df_old['direction_id'].astype(str) + '_' + + df_old['stop_id']

rt_dir_stop_new = set(df_new.route_dir_stop.unique().tolist())
rt_dir_stop_old = set(df_old.route_dir_stop.unique().tolist())

print(rt_dir_stop_old - rt_dir_stop_new)
print(len(list(rt_dir_stop_old - rt_dir_stop_new)))

#%% 
# Find the traversal time between stops for a single representative trip, based on scheduled departure time of the stop
#trips_interval = all_trips.copy()
trips = feed.trips[['trip_id', 'route_id', 'direction_id']]
# Filter trips df by trips_interval
trips = trips[trips.trip_id.isin(all_trips)]

trips_stoptimes = pd.merge(trips, feed.stop_times, on='trip_id', how='inner')[
    ['trip_id', 'route_id','direction_id','stop_sequence','stop_id','departure_time']]
trips_stoptimes = trips_stoptimes.sort_values(by='trip_id').groupby(
    by=['route_id', 'direction_id', 'stop_id', 'stop_sequence']).first().reset_index()
# trips_stoptimes = trips_stoptimes.sort_values(by='trip_id').groupby(
#     by=['route_id', 'direction_id', 'stop_sequence']).first().reset_index()
trips_stoptimes.sort_values(by=['route_id', 'direction_id', 'trip_id', 'stop_sequence'], ascending=True, inplace=True)

traversal_time_sec = trips_stoptimes.groupby(['route_id', 'direction_id'])['departure_time'].diff()  # minutes
trips_stoptimes['traversal_time_sec'] = traversal_time_sec

trips_stoptimes.to_csv(os.path.join(cwd, 'Data', 'Output_Data', 'PT_traversal_time.csv'))  
'''    
      
