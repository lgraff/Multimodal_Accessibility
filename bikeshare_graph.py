#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 11:22:59 2022

@author: lindsaygraff
"""

import geopandas as gpd
import pandas as pd
import os
import networkx as nx 
import util_functions as ut
import config as conf
#from create_study_area import study_area_gdf

# Inputs: bike network graph, filepath of bikeshare depot locations and attributes, lat/long/id/station name/availability
# column names in the file. 'availability' refers to the number of bike racks at the station
# Outputs: bikeshare graph inclusive of depot nodes and road intersection nodes. the depot nodes are connected to
# the intersection nodes by 'connection edges' formed by matching the depot to its nearest neighbor in the road network
def build_bikeshare_graph(G_bike, depot_filepath, lat_colname, long_colname, 
                          id_colname, name_colname, availability_colname, num_time_intervals, gdf_bike_network_nodes):
    G_bs = G_bike.copy()  
    nx.set_node_attributes(G_bs, 'bs', 'nwk_type')
    nx.set_node_attributes(G_bs, 'bs', 'node_type')
    nx.set_edge_attributes(G_bs, 'bs', 'mode_type')
    G_bs = ut.rename_nodes(G_bs, 'bs')
    
    # add price (which is time-independent)
    for e in G_bs.edges:
        avg_TT_sec =  G_bs.edges[e]['length_m'] / conf.config_data['Speed_Params']['bike'] 
        price = conf.config_data['Price_Params']['bs']['ppmin'] * (avg_TT_sec/60)  # op cost per edge (which is 0)
##        price_attr = dict(zip(['interval'+str(i)+'_price' for i in range(num_time_intervals)], 
##                              num_time_intervals * [price]))
        nx.set_edge_attributes(G_bs, {e:  {'0_price':price}})
        
        discomf = conf.config_data['Discomfort_Params']['bs'] #* avg_TT_min
        #discomf_attr = dict(zip(['interval'+str(i)+'_discomfort' for i in range(num_time_intervals)], num_time_intervals * [discomf]))
        nx.set_edge_attributes(G_bs, {e: {'0_discomfort':discomf}})

        
    # read in bikeshare depot locations and build connection edges
    df_bs = pd.read_csv(depot_filepath)
    # generate point geometry from x,y coords, so that the GIS clip function can be used to only include depots within the study region
    df_bs['geometry'] = gpd.points_from_xy(df_bs[long_colname], df_bs[lat_colname], crs="EPSG:4326")
    gdf_bs = gpd.GeoDataFrame(df_bs)  # convert to geo df
    gdf_bs['pos'] = tuple(zip(gdf_bs[long_colname], gdf_bs[lat_colname]))  # add position
    # Clip the bs node network
    study_area_gdf = gpd.read_file(os.path.join(os.path.join(os.getcwd(), 'Data', 'Output_Data'), 'study_area.csv'))
    gdf_bs_clip = gpd.clip(gdf_bs, study_area_gdf).reset_index().drop(columns=['index']).rename(
        columns={id_colname: 'id'})
    # join depot nodes and connection edges to the bikeshare (biking) network
    
    G_bs = ut.add_depots_cnx_edges(gdf_bs_clip, gdf_bike_network_nodes, # ['ID', name_colname, 'pos', availability_colname], 
                                   'bsd', 'bs', 'bs', num_time_intervals, G_bs, 'both')   
    for n in G_bs.nodes:
        if n.startswith('bsd'):
            G_bs.nodes[n]['node_type'] = 'bsd'
            G_bs.nodes[n]['nwk_type'] = 'bs'
    
    return G_bs


#in the function: add an argument of "cnx_edge_movement_type" 

# def add_depots_cnx_edges(gdf_depot_nodes, gdf_ref_nodes, depot_cols_keep, depot_id_prefix, 
#                          ref_id_prefix, movement_speed, num_intervals, G_ref, is_twoway_cnx=False):
#     # inputs: 
#     # output: 
        
#     # get point in reference network nearest to each parking node; only keep the ID of the point of the length 
#     # of the segment connecting the parking node to its nearest neighbor
#     nn = nearest_neighbor(gdf_depot_nodes, gdf_ref_nodes, 'y', 'x', return_dist=True)[['nn_osmid', 'length']]
#     #cols_keep = ['ID', 'pos', 'zone', 'float_rate']
#     cols_keep = depot_cols_keep + ['nn_osmid', 'length']
#     gdf_depot_nodes = pd.concat([gdf_depot_nodes, nn], axis=1)[cols_keep]
#     gdf_depot_nodes['ID'] = gdf_depot_nodes.apply(lambda row: depot_id_prefix + str(int(row['ID'])), axis=1)

#     # build cnx edges
#     gdf_cnx_edges = gdf_depot_nodes[['ID', 'nn_osmid', 'length']]
#     #gdf_pv_parking_edges = gdf_parking_edges_clip.copy()
#     gdf_cnx_edges['nn_osmid'] = gdf_cnx_edges.apply(lambda row: ref_id_prefix + str(int(row['nn_osmid'])), axis=1)
#     cnx_attr = (gdf_cnx_edges['length'] / movement_speed / 60).rename('avg_TT_min')  # m / (m/s) / (60s/min)
    
#     cnx_attr = pd.concat([cnx_attr] * 3*num_intervals, axis=1)
#     cnx_attr.columns = (['interval'+str(i) + '_' + 'avg_TT_min' for i in range(num_intervals)] + 
#                         ['interval'+str(i) + '_' + 'reliability' for i in range(num_intervals)] +
#                         ['interval'+str(i) + '_' + 'risk' for i in range(num_intervals)])
    
#     # add price
#     price_col_one_interval = pd.DataFrame([0] * len(gdf_cnx_edges))
#     price_col_all_intervals = pd.concat([price_col_one_interval] * num_intervals, axis=1)
#     price_col_all_intervals.columns = ['interval'+str(i) + '_' + 'price' for i in range(num_intervals)]
    
#     # add discomfort 
    
#     gdf_cnx_edges = pd.concat([gdf_cnx_edges, cnx_attr], axis=1)
#     gdf_cnx_edges.set_index(['nn_osmid', 'ID'], inplace=True)
#     cnx_edge_dict = gdf_cnx_edges.to_dict(orient='index')

#     # add connection edges to the graph. then add nodes and their attributes (position, zone name, rate)
#     G_ref.add_edges_from(list(cnx_edge_dict.keys()))
#     nx.set_edge_attributes(G_ref, cnx_edge_dict)
#     gdf_depot_nodes.set_index(['ID'], inplace=True)
#     node_dict = gdf_depot_nodes.drop(columns=['nn_osmid', 'length']).to_dict(orient='index')
#     nx.set_node_attributes(G_ref, node_dict)   
    
    
#     if is_twoway_cnx:
#         #print('************88')
#         oneway_edges = list(zip(*list(cnx_edge_dict.keys())))
#         other_edges = list(zip(oneway_edges[1], oneway_edges[0]))
#        # print(oneway_edges)
#         #print('------------')
#         #print(other_edges)
#         other_edges_attr = dict(zip(other_edges, cnx_edge_dict.values()))
#         G_ref.add_edges_from(list(other_edges))
#         nx.set_edge_attributes(G_ref, other_edges_attr)
        
#     return G_ref
    
