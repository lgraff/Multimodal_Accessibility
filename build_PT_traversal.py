#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 13:49:00 2022

@author: lindsaygraff
"""

import partridge as ptg
import pandas as pd
import os
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
trips_interval = feed.stop_times[
    (feed.stop_times.arrival_time >= config_data['Time_Intervals']['time_start']*60*60) # time_start = 7am
    & (feed.stop_times.arrival_time <= config_data['Time_Intervals']['time_end']*60*60) # time_end = 9am
    ].trip_id.unique().tolist()  # list of trip_ids in the time interval

# Find the traversal time between stops for a single representative trip, based on scheduled departure time of the stop
trips = feed.trips[['trip_id', 'route_id', 'direction_id']]
# Filter trips df by trips_interval
trips = trips[trips.trip_id.isin(trips_interval)]

trips_stoptimes = pd.merge(trips, feed.stop_times, on='trip_id', how='inner')[
    ['trip_id', 'route_id','direction_id','stop_sequence','stop_id','departure_time']]
trips_stoptimes = trips_stoptimes.sort_values(by='trip_id').groupby(
    by=['route_id', 'direction_id', 'stop_sequence']).first().reset_index()

# drop duplicates by route_id, direction_id, and stop_id triple i.e. focus on one trip
trips_stoptimes = trips_stoptimes.sort_values(by=['route_id', 'direction_id', 'stop_id', 'departure_time'], ascending=True)
trips_stoptimes.drop_duplicates(subset=['route_id','direction_id','stop_id'], inplace=True)

# trips_stoptimes = trips_stoptimes.sort_values(by='trip_id').groupby(
#     by=['route_id', 'direction_id', 'stop_sequence']).first().reset_index()
trips_stoptimes.sort_values(by=['route_id', 'direction_id', 'stop_sequence'], ascending=True, inplace=True)



traversal_time_sec = trips_stoptimes.groupby(['route_id', 'direction_id', 'trip_id'])['departure_time'].diff()  # minutes
trips_stoptimes['traversal_time_sec'] = traversal_time_sec

# remove the 28x bus route (this is user knoweldge, since the 28x only goes to airport and cannot be used for intracity travel)
trips_stoptimes = trips_stoptimes[trips_stoptimes['route_id'] != '28X']

trips_stoptimes.to_csv(os.path.join(cwd, 'Data', 'Output_Data', 'PT_traversal_time.csv'))            

