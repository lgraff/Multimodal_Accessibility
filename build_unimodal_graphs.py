#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 17:52:13 2022

@author: lindsaygraff
"""

# import libraries
import yaml
import geopandas as gpd
import pandas as pd
import os
import networkx as nx 
import matplotlib.pyplot as plt
from util_functions import *
from build_PT_graph import build_PT_graph
from build_bikeshare_graph import build_bikeshare_graph

# TO DO: in the original osm file, convert from multigraph to digraph
# this will eliminate repetitive keys 

# load config file
#config_data = load_config('config.yaml')
    
# read study area file
cwd = os.getcwd()
filepath = os.path.join(cwd, 'Data', 'Output_Data')
study_area_gdf = gpd.read_file(os.path.join(filepath, 'pgh_study_area.csv'))

# read edge files, node files, and crash files
gdf_drive_edges = gpd.read_file(os.path.join(filepath, 'gdf_safety_edges_veh.csv')).sort_values(by=['u','v','avg_TT_min']).drop_duplicates(['u','v'])
gdf_drive_nodes = gpd.read_file(os.path.join(filepath, 'osm_drive_nodes.csv'))
gdf_bike_edges = gpd.read_file(os.path.join(filepath, 'gdf_safety_edges_bike.csv')).sort_values(by=['u','v','avg_TT_min']).drop_duplicates(['u','v'])
gdf_bike_nodes = gpd.read_file(os.path.join(filepath, 'osm_bike_nodes.csv'))

# add risk index
gdf_drive_edges['crash_count'] = gdf_drive_edges['crash_count'].fillna(0)
gdf_bike_edges['crash_count'] = gdf_bike_edges['crash_count'].fillna(0)

# 1) drive risk: depends only on crash; 2) bike risk: depends on bike infrastructure
gdf_drive_edges.loc[:,'crash_per_meter'] = (gdf_drive_edges['crash_count'] / gdf_drive_edges['length'])
gdf_drive_edges.loc[:,'risk_idx'] = 1 + config_data['Risk_Parameters']['crash_weight'] * gdf_drive_edges['crash_per_meter']
gdf_bike_edges['risk_idx'] = gdf_bike_edges.apply(lambda row: calc_bike_risk_index(row), axis=1)


# Here we build the travel time multiplier as a function of time 
# some arbitary piecewise function
# add travel time by interval
len_period = int(config_data['Time_Intervals']['len_period'])
num_intervals = int(config_data['Time_Intervals']['len_period'] / config_data['Time_Intervals']['interval_spacing']) + 1

x = np.linspace(0, len_period, num_intervals )  # x is time [min past] relative to 07:00 AM
y = np.piecewise(x, [x < 15, ((x>=15) & (x<45)), ((x>=45) & (x<=75)), ((x>75) & (x<105)), x >= 105],
                 [1, lambda x: (0.5/30)*(x - 30) + 1.25, 1.5, lambda x: (-0.5/30)*(x - 90) + 1.25 , 1])
plt.plot(x, y, 'o', color='black', zorder=2);
plt.plot(x, y, color='red', zorder=1);
plt.xlabel('Time (minutes relative to 07:00AM)')
plt.ylabel('Travel time multiplier \n (relative to baseline)')


for i in range(num_intervals):
    # These can all be updated with real data as it is available 
    # travel time
    gdf_drive_edges['interval' + str(i) + '_avg_TT_min'] = y[i] * gdf_drive_edges['avg_TT_min']
    gdf_bike_edges['interval' + str(i) + '_avg_TT_min'] = gdf_bike_edges['avg_TT_min']
    # reliability
    gdf_drive_edges['interval' + str(i) + '_reliability'] = config_data['Active_Mode_Parameters']['rel_weight_nonactive'] * gdf_drive_edges['interval' + str(i) + '_avg_TT_min']
    gdf_bike_edges['interval' + str(i) + '_reliability'] = config_data['Active_Mode_Parameters']['rel_weight_active'] * gdf_bike_edges['interval' + str(i) + '_avg_TT_min']
    # risk
    gdf_drive_edges['interval' + str(i) + '_risk'] = gdf_drive_edges['risk_idx'] * gdf_drive_edges['avg_TT_min']
    gdf_bike_edges['interval' + str(i) + '_risk'] = gdf_bike_edges['risk_idx'] * gdf_bike_edges['avg_TT_min']
    # discomfort

# build nx graphs, complete with safety info (to add: elevation?)
interval_cols = [x for x in gdf_drive_edges.columns.tolist() if x.startswith('interval')]
edge_attr_cols = ['highway', 'length', 'speed_kph', 'travel_time', 'avg_TT_min',
       'tot_inj_sum', 'crash_count', 'geometry', 'risk_idx'] + interval_cols
G_drive = gdfs_to_nxgraph(gdf_drive_edges, gdf_drive_nodes, 'u', 'v', 'osmid', 'y', 'x', edge_attr_cols)
G_bike = gdfs_to_nxgraph(gdf_bike_edges, gdf_bike_nodes, 'u', 'v', 'osmid', 'y', 'x', edge_attr_cols + ['bikeway_type'])

# draw driving graph for visualization
# node_cmap = {'intersection node': '#1f77b4'}
# ax = draw_graph(G_drive, '#1f77b4', node_cmap, edge_color='gray')
# ax.set_title('Driving Network for Selected Neighborhoods', fontsize=16)

#%%
# TNC graph
G_tnc = G_drive.copy()
nx.set_node_attributes(G_tnc, 't', 'nwk_type')
nx.set_edge_attributes(G_tnc, 't', 'mode_type')
#G_tnc = rename_nodes(G_tnc, 't')

# add TNC price
TNC_ppmile = config_data['Price_Params']['TNC_ppmile']
TNC_ppmin = config_data['Price_Params']['TNC_ppmin']
miles_in_km = config_data['Conversion_Factors']['miles_in_km']
for e in G_tnc.edges:
    for i in range(num_intervals):
        G_tnc.edges[e]['interval'+str(i)+'_price'] = (TNC_ppmin * G_tnc.edges[e]['interval'+str(i)+'_avg_TT_min'] +
                                                      TNC_ppmile * miles_in_km * G_tnc.edges[e]['length']/1000)
#%%
# PERSONAL VEHICLE graph
# add park & ride?
G_pv = G_drive.copy()  # the personal vehicle graph is a copy of the driving graph
G_pv = rename_nodes(G_pv, 'pv')
nx.set_node_attributes(G_pv, 'pv', 'nwk_type')
nx.set_edge_attributes(G_pv, 'pv', 'mode_type')

# add price
pv_ppmile = config_data['Price_Params']['pv_ppmile']
meters_in_mile = config_data['Conversion_Factors']['meters_in_mile']
for e in G_pv.edges:
    price = pv_ppmile * (G_pv.edges[e]['length'] / meters_in_mile)  # op cost per edge
    #G_pv.edges[e]['price'] = price
    price_attr = dict(zip(['interval'+str(i)+'_price' for i in range(num_intervals)], num_intervals * [price]))
    nx.set_edge_attributes(G_pv, {e: price_attr})

# join parking nodes and connection edges to the personal vehicle network
gdf_parking_nodes = gpd.read_file(os.path.join(filepath, 'parking_points.csv'))
gdf_parking_nodes_clip = gpd.clip(gdf_parking_nodes, study_area_gdf).reset_index().drop(columns='index')
gdf_parking_nodes_clip['pos'] = tuple(zip(gdf_parking_nodes_clip['longitude'], gdf_parking_nodes_clip['latitude']))  # add position
gdf_parking_nodes_clip.insert(0, 'ID', gdf_parking_nodes_clip.index)  # add ID to each parking node

G_pv = add_depots_cnx_edges(gdf_parking_nodes_clip, gdf_drive_nodes, ['ID','pos','zone','float_rate'],
                                'k', 'pv', config_data['Speed_Params']['parking_speed'], interval_cols, G_pv)

# plot for visualization
node_color = ['black' if n.startswith('pv') else 'blue' for n in G_pv.nodes]
edge_color = ['grey' if e[0].startswith('pv') and e[1].startswith('pv') else 'magenta' for e in G_pv.edges]
ax = draw_graph(G_pv, node_color, {'road intersection':'black', 'pnr':'blue'}, edge_color)
ax.set_title('Personal Vehicle Network')

#%% PERSONAL BIKE graph
G_pb = G_bike.copy()
G_pb = rename_nodes(G_pb, 'pb')
nx.set_node_attributes(G_pb, 'pb', 'nwk_type')
nx.set_edge_attributes(G_pb, 'pb', 'mode_type')

for e in G_pb.edges:
    price = config_data['Price_Params']['pb_ppmin'] * G_pb.edges[e]['avg_TT_min']  # op cost per edge (which is 0)
    price_attr = dict(zip(['interval'+str(i)+'_price' for i in range(num_intervals)], num_intervals * [price]))
    nx.set_edge_attributes(G_pb, {e: price_attr})

#%% BIKESHARE graph
G_bs = build_bikeshare_graph(G_bike, filepath, 'Latitude', 'Longitude', 
                             'Station #', 'Station Name', '# of Racks')

#%% PUBLIC TRANSIT graph
G_pt = build_PT_graph(os.path.join(cwd, 'Data', 'Input_Data', 'GTFS'),
                      os.path.join(cwd, 'Data', 'Output_Data', 'PT_headway_NEW.csv'), 
                      os.path.join(cwd, 'Data', 'Output_Data', 'PT_traversal_time.csv'))

# G_bs = G_bike.copy()  
# nx.set_node_attributes(G_bs, 'bs', 'nwk_type')
# nx.set_edge_attributes(G_bs, 'bs', 'mode_type')
# G_bs = rename_nodes(G_bs, 'bs')

# # read in bikeshare depot locations and build connection edges
# filepath = os.path.join(cwd, 'Data', 'Input_Data')
# filename = 'pgh_bikeshare_depot_q3_2021.csv'
# df_bs = pd.read_csv(os.path.join(filepath, filename))
# # generate point geometry from x,y coords, so that the GIS clip function can be used to only include depots within the study region
# df_bs['geometry'] = gpd.points_from_xy(df_bs.Longitude, df_bs.Latitude, crs="EPSG:4326")
# gdf_bs = gpd.GeoDataFrame(df_bs)  # convert to geo df
# gdf_bs['pos'] = tuple(zip(gdf_bs['Longitude'], gdf_bs['Latitude']))  # add position
# # Clip the bs node network
# gdf_bs_clip = gpd.clip(gdf_bs, study_area_gdf).reset_index().drop(columns=['index']).rename(columns={'Station #': 'ID'})

# # join depot nodes and connection edges to the bikeshare (biking) network
# G_bs = add_depots_cnx_edges(gdf_bs_clip, gdf_bike_nodes, ['ID','Station Name', 'pos', '# of Racks'], 
#                             'bsd', 'bs', config_data['Speed_Params']['bike_speed'], interval_cols, 
#                             G_bs, is_twoway=True)

# plot for visualization
# node_color = ['blue' if n.startswith('bsd') else 'black' for n in G_bs.nodes]
# edge_color = ['magenta' if (e[0].startswith('bsd') or e[1].startswith('bsd')) else 'gray' for e in G_bs.edges]
# ax = draw_graph(G_bs, node_color, {'road intersection':'black', 'bs depot':'blue'}, edge_color)
# ax.set_title('Bikeshare Network')



#%%
# join parking nodes and connection edges to the personal vehicle network
# gdf_parking_nodes = gpd.read_file(os.path.join(filepath, 'parking_points.csv'))
# gdf_parking_nodes_clip = gpd.clip(gdf_parking_nodes, study_area_gdf).reset_index().drop(columns='index')
# gdf_parking_nodes_clip['pos'] = tuple(zip(gdf_parking_nodes_clip['longitude'], gdf_parking_nodes_clip['latitude']))  # add position
# gdf_parking_nodes_clip.insert(0, 'ID', gdf_parking_nodes_clip.index)  # add ID to each parking node

# # get point in drive network nearest to each parking node; only keep the ID of the point of the length 
# # of the segment connecting the parking node to its nearest neighbor
# nn = nearest_neighbor(gdf_parking_nodes_clip, gdf_drive_nodes, 'y', 'x', return_dist=True)[['nn_osmid', 'length']]
# cols_keep = ['ID', 'pos', 'zone', 'float_rate', 'nn_osmid', 'length']
# gdf_parking_nodes_clip = pd.concat([gdf_parking_nodes_clip, nn], axis=1)[cols_keep]
# gdf_parking_nodes_clip['ID'] = gdf_parking_nodes_clip.apply(lambda row: 'k'+str(int(row['ID'])), axis=1)

# # build cnx edges
# gdf_parking_edges_clip = gdf_parking_nodes_clip[['ID', 'nn_osmid', 'length']]
# #gdf_pv_parking_edges = gdf_parking_edges_clip.copy()

# gdf_parking_edges_clip['nn_osmid'] = gdf_parking_edges_clip.apply(lambda row: 'pv'+str(int(row['nn_osmid'])), axis=1)
# movement_speed = 5  # mph; how fast you go when looking for traversing the cnx edge

# park_cnx_attr = (gdf_parking_edges_clip['length'] / meters_in_mile / movement_speed / 60).rename('avg_TT_min')
# park_cnx_attr = pd.concat([park_cnx_attr] * len(interval_cols), axis=1)
# park_cnx_attr.columns = interval_cols
# gdf_parking_edges_clip = pd.concat([gdf_parking_edges_clip, park_cnx_attr], axis=1)
# gdf_parking_edges_clip.set_index(['nn_osmid', 'ID'], inplace=True)
# cnx_edge_dict = gdf_parking_edges_clip.to_dict(orient='index')

# # add connection edges to the graph. then add nodes and their attributes (position, zone name, rate)
# G_pv.add_edges_from(list(cnx_edge_dict.keys()))
# nx.set_edge_attributes(G_pv, cnx_edge_dict)
# gdf_parking_nodes_clip.set_index(['ID'], inplace=True)
# node_dict = gdf_parking_nodes_clip.drop(columns=['nn_osmid', 'length']).to_dict(orient='index')
# nx.set_node_attributes(G_pv, node_dict)




#df_park_nodes = gdf_parking_clip[['ID','latitude','longitude']] 
# G_park = gdfnodes_to_nxgraph(gdf_parking_clip, 'ID', 'latitude', 'longitude')
# nx.set_node_attributes(G_park, 'k', 'nwk_type')
# #G_park = rename_nodes(G_park, 'k')

# # merge the intersection nodes with the pnr stations. still need to add cnx edges
# G_pv = nx.union_all([G_pv, G_park])




# convert to digraph to eliminate redundant edge pairs
# G_drive = ox.utils_graph.get_digraph(G_drive, weight='travel_time')
# G_bike = ox.utils_graph.get_digraph(G_bike, weight='travel_time')