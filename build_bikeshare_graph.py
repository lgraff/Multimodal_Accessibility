#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 11:22:59 2022

@author: lindsaygraff
"""

import yaml
import geopandas as gpd
import pandas as pd
import os
import networkx as nx 
from util_functions import *

# Inputs: bike network graph, filepath of bikeshare depot locations and attributes, lat/long/id/station name/availability
# column names in the file. 'availability' refers to the number of bike racks at the station
# Outputs: bikeshare graph inclusive of depot nodes and road intersection nodes. the depot nodes are connected to
# the intersection nodes by 'connection edges' formed by matching the depot to its nearest neighbor in the road network
def build_bikeshare_graph(G_bike, depot_filepath, lat_colname, long_colname, 
                          id_colname, name_colname, availability_colname):
    G_bs = G_bike.copy()  
    nx.set_node_attributes(G_bs, 'bs', 'nwk_type')
    nx.set_edge_attributes(G_bs, 'bs', 'mode_type')
    G_bs = rename_nodes(G_bs, 'bs')
    
    # read in bikeshare depot locations and build connection edges
    df_bs = pd.read_csv(os.path.join(depot_filepath))
    # generate point geometry from x,y coords, so that the GIS clip function can be used to only include depots within the study region
    df_bs['geometry'] = gpd.points_from_xy(df_bs[long_colname], df_bs[lat_colname], crs="EPSG:4326")
    gdf_bs = gpd.GeoDataFrame(df_bs)  # convert to geo df
    gdf_bs['pos'] = tuple(zip(gdf_bs[long_colname], gdf_bs[lat_colname]))  # add position
    # Clip the bs node network
    gdf_bs_clip = gpd.clip(gdf_bs, study_area_gdf).reset_index().drop(columns=['index']).rename(
        columns={id_colname: 'ID'})

    # join depot nodes and connection edges to the bikeshare (biking) network
    G_bs = add_depots_cnx_edges(gdf_bs_clip, gdf_bike_nodes, ['ID', name_colname, 'pos', availability_colname], 
                                'bsd', 'bs', config_data['Speed_Params']['bike_speed'], interval_cols, 
                                G_bs, is_twoway_cnx=True)
    return G_bs
    
filepath = os.path.join(cwd, 'Data', 'Input_Data', 'pgh_bikeshare_depot_q3_2021.csv')
G_bs = build_bikeshare_graph(G_bike, filepath, 'Latitude', 'Longitude', 
                             'Station #', 'Station Name', '# of Racks')