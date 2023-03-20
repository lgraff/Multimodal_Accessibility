
#%% libraries
import numpy as np
import partridge as ptg
import pandas as pd
import os
from util_functions import *
import config as conf

config_data = conf.config_data

#%%
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
start_time = config_data['Time_Intervals']['time_start']*60*60
end_time = config_data['Time_Intervals']['time_end']*60*60

# %%
df_trips_stops = feed.stop_times.merge(feed.trips, how='inner', on='trip_id')
# buffer the end time by 1 hr so we can still calculate the headway for traveler arrivals at stops at the end of the study period
df_trips_stops = df_trips_stops[df_trips_stops['departure_time'].between(start_time, end_time + 1*60*60)]
# get dept times every 5 min
hour_range = [config_data['Time_Intervals']['time_start'], config_data['Time_Intervals']['time_end']-1]
min_range = [m for m in range(0,60,5)]
all_arrival_times_at_stop = [hr*60*60 + min*60 for hr in hour_range for min in min_range]

# routes = ['61C', '64']  # for testing
df_headway_rows = []
for r in all_routes: 
    for d in all_dirs: 
        # Get the stop_ids associated with the route-dir pair
        route_dir_condition = (df_trips_stops['route_id'] == r) & (df_trips_stops['direction_id'] == d)
        stop_ids = df_trips_stops[route_dir_condition].stop_id.unique().tolist()
        # Get headway for every stop
        for stop in stop_ids:
            condition = (df_trips_stops['route_id'] == r) & (df_trips_stops['direction_id'] == d) & (df_trips_stops['stop_id'] == stop)
            stop_times_filtered = df_trips_stops[condition][['route_id', 'direction_id', 'trip_id', 'stop_id', 'arrival_time', 'departure_time']].sort_values(by='departure_time', ascending=True)
            # Account for different traveler arrival times at the stop
            for a in all_arrival_times_at_stop:
                stop_dep_times = stop_times_filtered.departure_time.unique()  # departure times for the stop 
                try:
                    next_departure = np.sort(stop_dep_times[stop_dep_times>=a])[0]
                    headway = next_departure - a  # in seconds
                except:
                    headway = 1*60*60  # headway is at least one hour
                #print(a,next_departure)
                row = [r, d, stop, a, headway]
                df_headway_rows.append(row)

df_headway = pd.DataFrame(df_headway_rows, columns=['route_id', 'direction_id', 'stop_id', 'traveler_arrival_time', 'headway']).sort_values(by=['route_id','direction_id','stop_id','traveler_arrival_time'])
df_headway.to_csv(os.path.join(cwd, 'Data', 'Output_Data', 'PT_headway.csv'))     

        #print('**********')
        #     # any stop for any trip in the time interval for the route-dir pair
        #     all_stops = set([stop for stop_list in trip_stops for stop in stop_list])
# %% testing
#subject_stopID = '10947'
#condition = (df_trips_stops['route_id'] == '61C') & (df_trips_stops['direction_id'] == 0) & (df_trips_stops['stop_id'] == subject_stopID)
# for a given arrival time, find how long to wait until next bus (for that route and dir), according to schedule
# stop_times_filtered = df_trips_stops[condition][['route_id', 'direction_id', 'trip_id', 'stop_id', 'arrival_time', 'departure_time']].sort_values(by='departure_time', ascending=True)
# arrival_time_at_stop = 7*60*60 + 30*60  # 7:30am in seconds from midnight
# stop_dep_times = stop_times_filtered.departure_time.unique()  # departure times for the stop 
# next_departure = np.sort(stop_dep_times[stop_dep_times>arrival_time_at_stop])[0]
# headway = next_departure - arrival_time_at_stop  # in seconds

