#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 17:06:08 2022

@author: lindsaygraff
"""

# import libraries
#import yaml
#import geopandas as gpd
import pandas as pd
import geopandas as gpd
import os
import networkx as nx 
import re
import matplotlib.pyplot as plt
from shapely import wkt
import config as conf
import util_functions as ut

# cwd = os.getcwd()

# # load config file
# config_data = load_config('config.yaml')

# # read study area file

# filepath = os.path.join(cwd, 'Data', 'Output_Data')
# study_area_gdf = gpd.read_file(os.path.join(filepath, 'pgh_study_area.csv'))

# cwd = os.getcwd()
# GTFS_filepath = os.path.join(cwd, 'Data', 'Input_Data', 'GTFS')
# headway_filepath = os.path.join(cwd, 'Data', 'Output_Data', 'PT_headway.csv')
# traversal_time_filepath = os.path.join(cwd, 'Data', 'Output_Data', 'PT_traversal_time.csv')

def build_PT_graph(GTFS_filepath, headway_filepath, traversal_time_filepath, streets_processed_path):
    cwd = os.getcwd()
    # first we need the coordinates of all the bus stops, read directly from GTFS
    stops_df = pd.read_csv(os.path.join(GTFS_filepath, 'stops.txt'))
#**********************
    # Preprocessing: join bus stops to streets (nearest). 
    # read in stops and processed streets
    df_streets = pd.read_csv(streets_processed_path)
    df_streets['geometry'] = df_streets['geometry'].apply(wkt.loads)
    #df_streets[df_streets['pred_crash'].isna()] # quick check
    # convert to gdf for spatial join
    streets_gdf = gpd.GeoDataFrame(df_streets, geometry=df_streets.geometry, crs='EPSG:2272') 
    stops_gdf = gpd.GeoDataFrame(stops_df, geometry=gpd.points_from_xy(x=stops_df['stop_lon'], y=stops_df['stop_lat']), crs='EPSG:4326')
    # reproject to projected CRS
    streets_gdf.to_crs(crs='epsg:32128', inplace=True)
    stops_gdf.to_crs(crs='epsg:32128', inplace=True)
    # retain copy of streets geom for checking
    streets_gdf['saved_geom'] = streets_gdf.geometry
    # spatial join
    stops_streets = gpd.sjoin_nearest(stops_gdf, streets_gdf)[['stop_id','stop_lat','stop_lon','geometry','saved_geom','pred_crash']]
#**********************
    # add 'ps' in front of the stop_id to define it as a physical stop
    stops_df['stop_id'] = 'ps' + stops_df['stop_id']
    # add position
    stops_df['pos'] = tuple(
        zip(stops_df['stop_lon'], stops_df['stop_lat']))  
    stops_df.set_index('stop_id', inplace=True)
    cols_keep = ['stop_name', 'pos']
    # create dict of the form stop_id: {attr_name: attr_value}, then add to G_pt
    stopnode_dict = stops_df[cols_keep].to_dict(orient='index')
    G_pt = nx.DiGraph()
    G_pt.add_nodes_from(list(stopnode_dict.keys()))
    nx.set_node_attributes(G_pt, stopnode_dict)  
    nx.set_node_attributes(G_pt, 'ps', 'node_type') 
    nx.set_node_attributes(G_pt, 'pt', 'nwk_type') 

    #ax = ut.draw_graph(G_pt, 'blue', {'phsyical stop': 'blue'}, 'grey', 'solid')  # checks out

    # Convert to gdf, but don't clip to study area. Might be the case where the stops on a route go outside the study
    # area but then come back into the study area. If we remove the stops outside the study area, bus route will be
    # inconsistent with reality
    # stops_df['geometry'] = gpd.points_from_xy(
    #     stops_df.stop_lon, stops_df.stop_lat, crs='EPSG:4326')
    # stops_gdf = gpd.GeoDataFrame(stops_df, crs='EPSG:4326').rename(
    #     columns={'stop_lat': 'y', 'stop_lon': 'x'})


    # read headway and traversal time dfs
    # headway: how long between trips for a given route-dir pair
    # traversal time: how long from one stop_id to the next stop_id in the sequence for a given route-dir pair
    df_traversal_time = pd.read_csv(traversal_time_filepath)
    df_headway = pd.read_csv(headway_filepath)

    df_traversal_time[['stop_id','route_id','direction_id']] = df_traversal_time[['stop_id','route_id','direction_id']].astype('str')
    df_headway[['stop_id','route_id','direction_id']] = df_headway[['stop_id','route_id','direction_id']].astype('str')
    
    # define route_node_id as 'rt' + stop_id + route_id + dir_id
    df_headway['route_node_id'] = 'rt' + df_headway['stop_id'] + '_'+ df_headway['route_id']+ '_' + df_headway['direction_id']
    df_traversal_time['route_node_id'] = 'rt' + df_traversal_time['stop_id'] + '_' + df_traversal_time['route_id'] + '_' + df_traversal_time['direction_id'] + '_' + df_traversal_time['stop_sequence'].astype(str)

    # associate a route node to a stopID
    stops_df.reset_index(inplace=True)
    stops_df['stop_id'] = stops_df['stop_id'].str.replace('ps','')
    route_nodes_df = df_traversal_time.merge(
        stops_df, how='left', on='stop_id')[
            ['route_id', 'direction_id', 'stop_id', 'route_node_id', 'stop_sequence', 'pos']]
    route_nodes_df.set_index('route_node_id', inplace=True)
    route_node_dict = route_nodes_df.to_dict(orient='index')
    
    # build route edges
    # df_ss gives us the stops (as a list) associated with a route-dir pair
    df_ss = df_traversal_time.groupby(['route_id', 'direction_id']).agg(
         {'stop_id': list, 'stop_sequence': list}).reset_index()
    df_ss['id_seq'] = df_ss.apply(lambda x: list(zip(x.stop_id, x.stop_sequence)), axis=1)

      #df_ss.head(2)
      #df_traversal_time.loc[(df_traversal_time.route_id == "64") & (df_traversal_time.direction_id == 0)].head(3)

      # Build route edges
      # list of route-dir pairs
    route_dir_id_list = list(zip(df_ss.route_id, df_ss.direction_id))

    for i, s in enumerate(df_ss.id_seq):   # s is a list of (stop_id, stop_seq) tuples
        #stop_ids = df_ss.stop_id  #list(zip(*s))[0]  # list of sequential stop IDs along the route
        route_nodes = ['rt'+stop_id + '_' + route_dir_id_list[i][0] + '_' +
                       str(route_dir_id_list[i][1]) + '_' + str(stop_seq) for stop_id, stop_seq in s]
        # build route edge of the form: "rt" + stop_id + route_id + dir_id + stop_seq_num
        route_edges = list(
            zip(route_nodes[:len(route_nodes)], route_nodes[1:len(route_nodes)+1]))
        route_edges_attr = []
        # add travel time attribute. need to look it up in the df_traversal_time dataframe
        for e in route_edges:
            stop_id = e[0].split('rt')[1].split('_')[0]  # some str.split magic to get stop_id
            pred_crashes = stops_streets[stops_streets['stop_id'] == stop_id]['pred_crash'].values[0]
            trav_time_sec = df_traversal_time.loc[df_traversal_time['route_node_id'] == e[1]]['traversal_time_sec'].values[0]   # traversal time from GTFS data
            attr_dict = {'avg_TT_sec': trav_time_sec, 'pred_crash':pred_crashes}
            # tt_attr = {'0_avg_TT_sec': trav_time_sec}
            # price_attr = {'0_price': 0}  # the 2.75 boarding cost will be embedded in boarding edge
            # reliability_attr = {'0_reliability': conf.config_data['Reliability_Params']['PT_traversal'] * trav_time_sec}
            # risk_attr = {'0_risk': conf.config_data['Risk_Parameters']['PT_traversal']}
            # discomf_attr = {'0_discomfort': conf.config_data['Discomfort_Params']['PT_traversal']}
            route_edges_attr.append((e[0], e[1], attr_dict))  # | is an operator for merging dicts

        # add route edges to the PT graph, along with attriutes
        G_pt.add_edges_from(route_edges_attr)

    nx.set_edge_attributes(G_pt, "pt", 'mode_type')
    nx.set_node_attributes(G_pt, route_node_dict)
    nx.set_node_attributes(G_pt, {r: {'nwk_type':'pt', 'node_type':'rt'} for r in route_node_dict.keys()}) 
    #nx.set_node_attributes(G_rt, 'rt', 'nwk_type')

        # add boarding and alighting edges. avg_TT_min is the waiting time
        # add time to board and time to alight? just a few seconds, so will be negligible probably
        
    ba_edges = []
    for n in list(G_pt.nodes):
        if G_pt.nodes[n]['node_type'] == 'rt':  #if n.startswith('rt'):   # is a route node
            # Find associated physical stop
            split_route_node = n.split('_')
            #r, d, stop = # FILL IN HERE  split_route_node[0].split('rt')[1]
            #phys_stop = 'ps' + stop 
            
            # re.sub('\D', '', string) removes letters from string
            phys_stop = 'ps' + re.sub('\D', '', (split_route_node[0]))

            # Find headway associated with the route-dir-stop
            # remove stop sequence number
            #stop_route_dir_id = split_route_node[0] + '_' + \
             #   split_route_node[1] + '_' + split_route_node[2]
            #print(stop_route_dir_id)

            #headway_by_arrival_time = {}
            #df_headway_filtered = df_headway[(df_headway['route_id'] == r) & (df_headway['direction_id']==d) & (df_headway['stop_id']==stop)].sort_values(by='traveler_arrival_time')
            #headway_by_arrival_time = {[for h in df_headway_filtered.headway.values()]}
            # headway_min = df_headway.loc[df_headway['route_node_id'] == stop_route_dir_id][
            #     'headway_min'].values[0]  # headway in minutes

            # BOARDING edges
            e_board = (phys_stop, n)
            ba_edges.append((e_board[0], e_board[1], {'mode_type':'board'}))
            # add price of one-way fare and waiting cost as headway/2; # call waiting cost "avg_TT_min" for sake of consistency in attribute definitions
            #attr_dict = {'avg_TT_min': headway_min/2, 'price': 2.75}  # ,
            # add pure attributes
            # trav_time_sec = (headway_min*60) / 2 # avg wait time is defined as headway/2
            # tt_attr = {'0_avg_TT_sec': trav_time_sec}
            # price_attr = {'0_price': conf.config_data['Price_Params']['PT']['fixed']}  # the 2.75 boarding cost isembedded in boarding edge
            # reliability_attr = {'0_reliability': conf.config_data['Reliability_Params']['PT_wait'] * trav_time_sec}
            # risk_attr = {'0_risk': conf.config_data['Risk_Parameters']['PT_wait']}
            # discomf_attr = {'0_discomfort': conf.config_data['Discomfort_Params']['PT_wait']}
            # ba_edges.append((e_board[0], e_board[1],  tt_attr | price_attr | reliability_attr | risk_attr | discomf_attr | {'mode_type':'board'}))
            
            # ALIGHTING edges
            e_alight = (n, phys_stop)
            ba_edges.append((e_alight[0], e_alight[1], {'mode_type':'alight'}))
            # alighting edge has price = 0 and TT = 0 ?
            # note: we will need to use the node cost file to remove 2.75 when going rt-ps-ps-rt
            # for now, everything is 0
            # tt_attr = {'0_avg_TT_sec': 0}
            # price_attr = {'0_price': 0}
            # reliability_attr = {'0_reliability': 0}
            # risk_attr = {'0_risk': 0}
            # discomf_attr = {'0_discomfort': 0}
            # ba_edges.append((e_alight[0], e_alight[1], tt_attr | price_attr | reliability_attr | risk_attr | discomf_attr | {'mode_type':'alight'}))
            
            # offset the geometry of the route nodes, for visualization purposes
            G_pt.nodes[n]['pos'] = (G_pt.nodes[n]['pos'][0] + 0.001, G_pt.nodes[n]['pos'][1] + 0.001)

    G_pt.add_edges_from(ba_edges)  # add board/alight edges to the graph

    # offset the geometry of the route nodes, for visualization purposes
    # for n in G_pt.nodes:
    #     if G_pt.nodes[n]['node_type'] == 'rt': #n.startswith('rt'):
    #         G_pt.nodes[n]['pos'] = (
    #             G_pt.nodes[n]['pos'][0] + 0.001, G_pt.nodes[n]['pos'][1] + 0.001)
            
    return G_pt

#%% test the function
# cwd = os.getcwd()
# G_pt_full = build_PT_graph(os.path.join(cwd, 'Data', 'Input_Data', 'GTFS'),
#                       os.path.join(cwd, 'Data', 'Output_Data', 'PT_headway_NEW.csv'), 
#                       os.path.join(cwd, 'Data', 'Output_Data', 'PT_traversal_time.csv'))