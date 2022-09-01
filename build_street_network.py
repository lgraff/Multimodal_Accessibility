#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 16:08:26 2022

@author: lindsaygraff
"""

# import libraries
import geopandas as gpd
import pandas as pd
import os
import osmnx as ox
import networkx as nx

# read study area file
cwd = os.getcwd()
filepath = os.path.join(cwd, 'Data', 'Output_Data')
study_area_gdf = gpd.read_file(os.path.join(filepath, 'pgh_study_area.csv'))

# import drive and bike street networks
# retain_all = False argument means that the smaller disconnected network is not retained
G_drive = ox.graph_from_polygon(study_area_gdf['geometry'][0], network_type='drive', retain_all=False)
G_bike = ox.graph_from_polygon(study_area_gdf['geometry'][0], network_type='bike', retain_all=False)

# add edge speed and travel time
ox.speed.add_edge_speeds(G_drive)
ox.speed.add_edge_speeds(G_bike)
ox.speed.add_edge_travel_times(G_drive)  # in seconds
ox.speed.add_edge_travel_times(G_bike)  # in seconds

# convert nx graph to geodf
gdf_nodes_drive, gdf_edges_drive = ox.graph_to_gdfs(G_drive)
gdf_edges_drive.reset_index(inplace=True)
gdf_nodes_bike, gdf_edges_bike = ox.graph_to_gdfs(G_bike)
gdf_edges_bike.reset_index(inplace=True)

#G = ox.graph_from_gdfs(gdf_nodes, gdf_edges, graph_attrs=G.graph)

gdf_edges_drive['avg_TT_min'] = gdf_edges_drive['travel_time'] / 60
gdf_edges_bike['avg_TT_min'] = gdf_edges_bike['travel_time'] / 60

# save as individual files for edges and nodes
cols_keep = ['u','v', 'geometry', 'highway', 'length','speed_kph', 'travel_time','avg_TT_min']

#%%
# Remove edge duplicates (which were created b/c of multigraph structure)
# 1) For vehicle graph, keep edge that has shorter travel time
# 2) For bicycle graph, keep edge based on higway type hierarchy
# This is accomplished by exploding the highway column, then taking the more "bikeable" road
gdf_edges_drive = gdf_edges_drive.explode('highway')
gdf_edges_drive = gdf_edges_drive.sort_values(by=['u','v','avg_TT_min']).drop_duplicates(['u','v'])

gdf_edges_bike = gdf_edges_bike.explode('highway')
hierarchy = {'cycleway':0, 'path':1, 'service': 1.5, 'residential':2, 
             'unclassified':3,
             'tertiary':4, 'tertiary_link':5, 'secondary':6, 'secondary_link':7,
             'primary':8, 'primary_link':9}
gdf_edges_bike['new_highway'] = gdf_edges_bike['highway'].map(hierarchy)
gdf_edges_bike = gdf_edges_bike.sort_values(['u','v','new_highway']).drop_duplicates(['u','v'])


#%%
#gdf_edges_drive['highway'] = gdf_edges_drive['highway'].astype('str')
gdf_edges_drive[cols_keep].to_file(os.path.join(filepath, 'osm_drive_edges.csv'), driver='GeoJSON') #, index=False)
gdf_nodes_drive.to_file(os.path.join(filepath, 'osm_drive_nodes.csv'), driver='GeoJSON') #, index=False)
#gdf_edges_bike['highway'] = gdf_edges_bike['highway'].astype('str')
gdf_edges_bike[cols_keep].to_file(os.path.join(filepath, 'osm_bike_edges.csv'), driver='GeoJSON') #, index=False)
gdf_nodes_bike.to_file(os.path.join(filepath, 'osm_bike_nodes.csv'), driver='GeoJSON') #, index=False)

