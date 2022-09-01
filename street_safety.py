#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 18:43:35 2022

@author: lindsaygraff

street safety analysis
"""

# import libraries
import geopandas as gpd
import pandas as pd
import os
#import osmnx as ox
#import ckanapi
#import networkx as nx
from util_functions import *

# read study area file
cwd = os.getcwd()
filepath = os.path.join(cwd, 'Data', 'Output_Data')
study_area_gdf = gpd.read_file(os.path.join(filepath, 'pgh_study_area.csv'))

gdf_edges_drive = gpd.read_file(os.path.join(filepath, 'osm_drive_edges.csv'))
gdf_nodes_drive = gpd.read_file(os.path.join(filepath, 'osm_drive_nodes.csv'))
gdf_edges_bike = gpd.read_file(os.path.join(filepath, 'osm_bike_edges.csv'))
gdf_nodes_bike = gpd.read_file(os.path.join(filepath, 'osm_bike_edges.csv'))

# get last 2 years of crash data
site = "https://data.wprdc.org"
crash_data_2020 = get_resource_data(site,resource_id="514ae074-f42e-4bfb-8869-8d8c461dd824",count=999999999) 
crash_data_2019 = get_resource_data(site,resource_id="cb0a4d8b-2893-4d20-ad1c-47d5fdb7e8d5",count=999999999) 

# Convert to pandas df and concatenate
df_crash_2020 = pd.DataFrame(crash_data_2020)
df_crash_2019 = pd.DataFrame(crash_data_2019)
#list(df_crash_2020.columns)
cols_keep = ['DEC_LAT', 'DEC_LONG', 'BICYCLE', 'BICYCLE_COUNT', 'PEDESTRIAN', 'PED_COUNT', 
             'SPEED_LIMIT', 'VEHICLE_COUNT', 'TOT_INJ_COUNT']
df_crash_2020 = df_crash_2020[cols_keep]
df_crash_2019 = df_crash_2019[cols_keep]
df_crash = pd.concat([df_crash_2019, df_crash_2020], ignore_index=True)
# Remove rows that do not have both a lat and long populated
df_crash = df_crash.loc[~((df_crash['DEC_LAT'].isnull()) | (df_crash['DEC_LONG'].isnull()))]
# Convert deg-min-sec to decimal degrees
gdf_crash = gpd.GeoDataFrame(df_crash, geometry=gpd.points_from_xy(x=df_crash['DEC_LONG'], y=df_crash['DEC_LAT']), 
                             crs='EPSG:4326')
# Clip to neighborhood mask
gdf_crash_clip = gpd.clip(gdf_crash, study_area_gdf)

# Separate crashes by bike, pedestrian, vehicle
gdf_ped_crash = gdf_crash_clip.loc[gdf_crash_clip.PEDESTRIAN == 1]  # pedestrian crashes
gdf_bike_crash = gdf_crash_clip.loc[gdf_crash_clip.BICYCLE == 1]  # bicycle crashes
gdf_veh_crash = gdf_crash_clip.loc[gdf_crash_clip.VEHICLE_COUNT > 1]  # vehicle crashes

# flatten projections
for g in [gdf_bike_crash, gdf_veh_crash, gdf_edges_bike, gdf_edges_drive]:
    g.to_crs(crs=3857, inplace=True)

# join crash to its nearest road segment edge 
def join_crash_to_edge(crash_gdf, edge_gdf):
    gdf_crash_edges = crash_gdf.sjoin_nearest(edge_gdf, how='inner', distance_col = 'Distance')
    #temp = gpd.sjoin_nearest(crash_gdf, edge_gdf)
    crash_grouped = gdf_crash_edges.groupby(['u','v']).agg({
        'TOT_INJ_COUNT':['sum','count']}).reset_index() #.sort_values(by='TOT_INJ_COUNT')
    crash_grouped.columns = ['u','v','tot_inj_sum', 'crash_count']
    crash_grouped = pd.merge(edge_gdf, crash_grouped, on=['u','v'], how='left')
    return crash_grouped

gdf_crash_edges_bike = join_crash_to_edge(gdf_bike_crash, gdf_edges_bike)
gdf_crash_edges_veh = join_crash_to_edge(gdf_veh_crash, gdf_edges_drive)

#%% Bikelane
filepath = os.path.join(cwd, 'Data', 'Input_Data', 'bike-map-2019')

# ** TO DO** Add more bikeway types

bikeway_type = ['Bike Lanes', 'On Street Bike Route', 'Protected Bike Lane']
gdf_bikeway = gpd.GeoDataFrame()
for b in bikeway_type:
    new_path = os.path.join(filepath, b)
    filename = b + '.shp'
    gdf =  gpd.read_file(os.path.join(new_path, filename))
    gdf.to_crs(crs = 4326, inplace=True)
    gdf['bikeway_type'] = b
    cols_keep = ['geometry','bikeway_type']
    gdf = gdf[cols_keep]
    gdf_bikeway = pd.concat([gdf_bikeway, gdf])
gdf_bikeway['bikeway_type'].unique()
gdf_bikeway = gpd.clip(gdf_bikeway, study_area_gdf)

# join gdf_bikeway to gdf_bike_edges
# steps: buffer the bikeway edges -- creates a polygon
# sjoin  how=left, left_gdf = gdf_edges_bike, right_gdf = buffered_bikeway, predicate=intersect

#%%
# Buffer the network edges
gdf_crash_edges_bike['line_buffer_geom'] = gdf_crash_edges_bike['geometry'].buffer(distance = 2)  # 2 meter radius
gdf_crash_edges_bike.set_geometry('line_buffer_geom', inplace=True)
gdf_crash_edges_bike.drop_duplicates(['u','v'], inplace=True)

# gdf_crash_edges_bike.plot()

gdf_bikeway.to_crs(crs=3857, inplace=True)
temp = gpd.GeoDataFrame(gpd.sjoin(gdf_crash_edges_bike, gdf_bikeway, how='left', predicate='intersects'))
temp.set_geometry('geometry', inplace=True)

hierarchy = {'Protected Bike Lane':0, 'Bike Lanes':1, 'On Street Bike Route':2}
temp['new_bikeway_type'] = temp['bikeway_type'].map(hierarchy)
gdf_bike_crash_bikeway = temp.sort_values(['u','v','new_bikeway_type']).drop_duplicates(['u','v'])

#%% plot for visualization. see if this method worked
bikelane_approx = gdf_bike_crash_bikeway.loc[~gdf_bike_crash_bikeway.bikeway_type.isna()]

fig, ax = plt.subplots()
bikelane_approx.plot(ax=ax, color='blue')
gdf_bikeway.plot(ax=ax, color='green', alpha=0.3)
# we see that we got more edges than are actually represented by bike lanes
# but it's relatively accurate

#%% save as csv files
filepath = os.path.join(cwd, 'Data', 'Output_Data')
gdf_crash_edges_veh.to_file(os.path.join(filepath, 'gdf_safety_edges_veh.csv'), driver='GeoJSON')
gdf_bike_crash_bikeway.drop(['new_bikeway_type', 'index_right', 'line_buffer_geom'], axis=1, inplace=True)
gdf_bike_crash_bikeway.to_file(os.path.join(filepath, 'gdf_safety_edges_bike.csv'), driver='GeoJSON')
