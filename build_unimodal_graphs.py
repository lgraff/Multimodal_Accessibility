#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 17:52:13 2022

@author: lindsaygraff
"""
#%%
# import libraries
import geopandas as gpd
import pandas as pd
import os
import networkx as nx 
import numpy as np
import matplotlib.pyplot as plt
import pickle
from shapely import wkt, geometry
import util_functions as ut
import config as conf
from PT_graph import build_PT_graph
from bikeshare_graph import build_bikeshare_graph
#from create_study_area import conf.study_area_gdf
#import global_ as gl
#from global_ import config_data, conf.study_area_gdf

# TO DO: in the original osm file, convert from multigraph to digraph
# this will eliminate repetitive keys 

# load config file
#config_data = load_config('config.yaml')
    
# # read study area file
study_area_gdf = gpd.read_file(os.path.join(os.path.join(os.getcwd(), 'Data', 'Output_Data'), 'study_area.csv'))
#study_area_gdf = conf.study_area_gdf
# some parameters
len_period = int(conf.config_data['Time_Intervals']['len_period'])
num_intervals = int(conf.config_data['Time_Intervals']['len_period'] / conf.config_data['Time_Intervals']['interval_spacing']) + 1

# conf.study_area_gdf = gpd.read_file(os.path.join(filepath, 'pgh_study_area.csv'))
# read graphs that were created in 'process_street_centerlines.py'
cwd = os.getcwd()
with open(os.path.join(cwd, 'Data', 'Output_Data', 'G_drive.pkl'), 'rb') as inp:
    G_drive = pickle.load(inp)  # can think about subsetting the attributes
with open(os.path.join(cwd, 'Data', 'Output_Data', 'G_bike.pkl'), 'rb') as inp:
    G_bike = pickle.load(inp)

#%%
# *idea*: let's only store the "base" travel time, i.e. at the 0th interval when TT multipler is equal to 1
# *i think* that all other values can be derived from this one
# DRIVE
# for e in G_drive.edges:
#     # travel time: avg_TT = TT_multiplier * (distance / speed_limit)
#     G_drive.edges[e]['0' + '_avg_TT_sec'] =  (G_drive.edges[e]['length_m'] / 
#                                                               conf.config_data['Conversion_Factors']['meters_in_mile'] /
#                                                               G_drive.edges[e]['speed_lim'] * 60 * 60)
#     # reliability
#     # (maybe: also evaluate road type i.e. residential roads may not have high reliability mult)
#     G_drive.edges[e]['0' + '_reliability'] = conf.config_data['Reliability_Params']['drive'] * G_drive.edges[e]['0' + '_avg_TT_sec']
#     # risk: we will remove the TT component
#     G_drive.edges[e]['0' + '_risk'] = G_drive.edges[e]['risk_idx_drive'] #* G_drive.edges[e]['interval' + str(i) + '_avg_TT_min']

# # BIKE
# for e in G_bike.edges:
#     # travel time: avg_TT = TT_multiplier * (distance / speed_limit)
#     G_bike.edges[e]['0' + '_avg_TT_sec'] =  (G_bike.edges[e]['length_m']/conf.config_data['Speed_Params']['bike'])  
#     # reliability
#     G_bike.edges[e]['0' + '_reliability'] = conf.config_data['Reliability_Params']['bike'] * G_bike.edges[e]['0' + '_avg_TT_sec']
#     # risk:  we will remove the TT component
#     G_bike.edges[e]['0' + '_risk'] = G_bike.edges[e]['risk_idx_bike'] #* G_bike.edges[e]['interval' + str(i) + '_avg_TT_min']
#%% create df_node files for both driving and biking

def create_gdf_nodes(G):    
    df = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')
    df[['long', 'lat']] = pd.DataFrame(df['pos'].tolist(), index=df.index)
    df['id'] = df.index
    df['id'] = df['id'].astype('int')
    df.drop(columns='pos', inplace=True)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.long, df.lat))
    gdf.set_crs(crs='epsg:4326', inplace=True)
    return gdf

gdf_drive_nodes = create_gdf_nodes(G_drive)
gdf_bike_nodes = create_gdf_nodes(G_bike)
# cwd = os.getcwd()

# # draw driving graph for visualization
# # To do: go back and ensure no self-loops
# node_cmap = {'intersection node': '#1f77b4'}
# ax = ut.draw_graph(G_drive, '#1f77b4', node_cmap, 'gray', 'solid')
# ax.set_title('Driving Network for Selected Neighborhoods', fontsize=16)

# TNC graph: 
    # Attributes: TT, reliability, risk, price, discomfort
G_tnc = G_drive.copy()
nx.set_node_attributes(G_tnc, 't', 'nwk_type')  
nx.set_node_attributes(G_tnc, 't', 'node_type')# all nodes have same node type (i.e. no special nodes)
nx.set_edge_attributes(G_tnc, 't', 'mode_type')
G_tnc = ut.rename_nodes(G_tnc, 't')

# add TNC price
# TNC_ppmile = conf.config_data['Price_Params']['TNC']['ppmile']
# TNC_ppmin = conf.config_data['Price_Params']['TNC']['ppmin']
# miles_in_km = conf.config_data['Conversion_Factors']['miles_in_km']
# for e in G_tnc.edges:
#     #for i in range(num_intervals):
#     G_tnc.edges[e]['0'+'_price'] = TNC_ppmile * miles_in_km * G_tnc.edges[e]['length_m']/1000   # $/mile * mile/km * km
#                                     # + (TNC_ppmin / 60) * G_tnc.edges[e]['0'+'_avg_TT_sec'] +                                     
#     G_tnc.edges[e]['0'+'_discomfort'] = conf.config_data['Discomfort_Params']['TNC']

# ** just for testing **
# G_tnc.add_nodes_from([('org', {'pos':(-79.94868171046522, 40.416379503934145)}),
#                       ('dst', {'pos':(-79.92070090793109, 40.463543819430086)})])

# plot for visualization
node_color = ['black' if n.startswith('t') else 'blue' for n in G_tnc.nodes]
edge_color = ['grey'] * len(list(G_tnc.edges))
ax = ut.draw_graph(G_tnc, node_color, {'road intersection':'black', 'o/d':'blue'}, edge_color, 'solid')
#ax.set_title('Personal Vehicle Network')

#%% PERSONAL VEHICLE graph
# add park & ride?
G_pv = G_drive.copy()  # the personal vehicle graph is a copy of the driving graph
G_pv = ut.rename_nodes(G_pv, 'pv')
nx.set_node_attributes(G_pv, 'pv', 'nwk_type')
nx.set_node_attributes(G_pv, 'pv', 'node_type')
nx.set_edge_attributes(G_pv, 'pv', 'mode_type')

# add price and dsicomfort
# meters_in_mile = conf.config_data['Conversion_Factors']['meters_in_mile']
# for e in G_pv.edges:
#     price = conf.config_data['Price_Params']['pv']['ppmile'] * (G_pv.edges[e]['length_m'] / meters_in_mile)  # op cost per edge
#     #G_pv.edges[e]['price'] = price
#     #price_attr = dict(zip(['interval'+str(i)+'_price' for i in range(num_intervals)], num_intervals * [price]))
#     #nx.set_edge_attributes(G_pv, {e: price_attr})
#     #for i in range(num_intervals):
#     G_pv.edges[e]['0'+'_price'] = price
#     G_pv.edges[e]['0'+'_discomfort'] = conf.config_data['Discomfort_Params']['pv']

# join parking nodes and connection edges to the personal vehicle network
filepath = os.path.join(cwd, 'Data', 'Output_Data')
gdf_parking_nodes = gpd.read_file(os.path.join(filepath, 'parking_points.csv'))
#gdf_parking_nodes_clip = gpd.clip(gdf_parking_nodes, conf.study_area_gdf).reset_index().drop(columns='index')
gdf_parking_nodes['pos'] = tuple(zip(gdf_parking_nodes['longitude'], gdf_parking_nodes['latitude']))  # add position
gdf_parking_nodes.insert(0, 'id', gdf_parking_nodes.index)  # add ID to each parking node

G_pv = ut.add_depots_cnx_edges(gdf_parking_nodes, gdf_drive_nodes, # ['ID','pos','zone','float_rate'],
                               'k', 'pv', 'pv', num_intervals, G_pv, 'to_depot')
# rename mode_type of parking edges
for e in G_pv.edges:
    if e[1].startswith('k'):
        G_pv.edges[e]['mode_type'] = 'park'

#plot for visualization
node_color = ['black' if n.startswith('pv') else 'blue' for n in G_pv.nodes]
edge_color = ['grey' if e[0].startswith('pv') and e[1].startswith('pv') else 'magenta' for e in G_pv.edges]
ax = ut.draw_graph(G_pv, node_color, {'road intersection':'black', 'pnr':'blue'}, edge_color, 'solid')
ax.set_title('Personal Vehicle Network')

#%% PERSONAL BIKE graph:
    # Attributes: TT, reliability, risk, price, discomfort
G_pb = G_bike.copy()
G_pb = ut.rename_nodes(G_pb, 'pb')
nx.set_node_attributes(G_pb, 'pb', 'nwk_type')
nx.set_node_attributes(G_pb, 'pb', 'node_type')
nx.set_edge_attributes(G_pb, 'pb', 'mode_type')

# for e in G_pb.edges:
#     avg_TT_sec =  G_pb.edges[e]['length_m'] / conf.config_data['Speed_Params']['bike'] 
#     price = conf.config_data['Price_Params']['pb']['ppmin'] * avg_TT_sec  # op cost per edge (which is 0)
#     #price_attr = dict(zip(['interval'+str(i)+'_price' for i in range(num_intervals)], num_intervals * [price]))
#     nx.set_edge_attributes(G_pb, {e: {'0_price':price}})

#     discomf = conf.config_data['Discomfort_Params']['pb'] #* avg_TT_min
#     #discomf_attr = dict(zip(['interval'+str(i)+'_discomfort' for i in range(num_intervals)], num_intervals * [discomf]))
#     nx.set_edge_attributes(G_pb, {e: {'0_discomfort':discomf}})

#%% BIKESHARE graph:
    # Attributes: TT, reliability, risk, price, discomfort
    # *except* connection edges do not yet have all 5
bs_filepath = os.path.join(cwd, 'Data', 'Input_Data', 'pogoh-station-locations-2022.csv') #'pgh_bikeshare_depot_q3_2021.csv')
G_bs = build_bikeshare_graph(G_bike, bs_filepath, 'Latitude', 'Longitude', 
                             'Id', 'Name', 'Total Docks', num_intervals, gdf_bike_nodes)
#plot for visualization
node_color = ['blue' if n.startswith('bsd') else 'black' for n in G_bs.nodes]
edge_color = ['grey' if e[0].startswith('bs') and e[1].startswith('bs') else 'magenta' for e in G_bs.edges]
ax = ut.draw_graph(G_bs, node_color, {'bike share station':'blue', 'road intersection':'black'}, edge_color, 'solid')
ax.set_title('Bike share Network')
# for n in G_bs.nodes:
#     if not 'pos' in G_bs.nodes[n]:
#         print(n)

#nodes without a position:
    #bs-94520590
    #bs-416224899
    #bs-1528424056

#%% PUBLIC TRANSIT graph
G_pt_full = build_PT_graph(os.path.join(cwd, 'Data', 'Input_Data', 'GTFS'),
                      os.path.join(cwd, 'Data', 'Output_Data', 'PT_headway_NEW.csv'), 
                      os.path.join(cwd, 'Data', 'Output_Data', 'PT_traversal_time.csv'))

# Reduce the size of the PT network through a bounding box approach:
# Find the bounding box of the pgh_study_area polygon. Extend this bounding box by x miles. Then clip the PT network by this extended bounding box

df = pd.DataFrame.from_dict(dict(G_pt_full.nodes), orient="index").reset_index()
#gdf_pt = gpd.GeoDataFrame(data=df, geometry=df.pos)
df[['x','y']] = pd.DataFrame(df.pos.tolist())
gdf_ptnodes = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x,df.y), crs='EPSG:4326')
gdf_ptnodes.head(3)
#bbox_study_area = conf.study_area_gdf['geometry'].bounds.T.to_dict()[0]  # bounding box of neighborhood polygon layer
#bbox_df = study_area_gdf['geometry'].bounds
x = 0.25 # miles (buffer the PT network even more than the street network b/c we can imagine the case of a bus route going outside the bounds and then returning inside)
study_area_buffer = study_area_gdf.to_crs(crs='epsg:32128').buffer(x*1609).to_crs('EPSG:4326')  # 1609 meters/mile
# check that this worked
fig,ax = plt.subplots(figsize=(4,4))
study_area_gdf.plot(ax=ax, color='blue')
study_area_buffer.plot(ax=ax, color='green', alpha=.4)

# # extend the bounds of the study area
# bbox_df['newminx'] = bbox_df['minx'] - 1/69 * x  # 1 degree/69 mile
# bbox_df['newmaxx'] = bbox_df['maxx'] + 1/69 * x
# bbox_df['newminy'] = bbox_df['miny'] - 1/69 * x 
# bbox_df['newmaxy'] = bbox_df['maxy'] + 1/69 * x
# # define new points that will be used to define an extended bounding box
# pt1 = geometry.Point(bbox_df.newminx, bbox_df.newminy)
# pt2 = geometry.Point(bbox_df.newminx, bbox_df.newmaxy)
# pt3 = geometry.Point(bbox_df.newmaxx, bbox_df.newmaxy)
# pt4 = geometry.Point(bbox_df.newmaxx, bbox_df.newminy)
# bbox_new = geometry.Polygon((pt1,pt2,pt3,pt4))
# bbox_new_gdf = gpd.GeoDataFrame(gpd.GeoSeries(bbox_new), columns=['geometry'])

# clip the list of all pt nodes to just those within the new bbox
pt_graph_clip = gpd.clip(gdf_ptnodes, study_area_buffer)
pt_graph_clip.set_index('index', inplace=True)
# go back to the original PT graph, only keep nodes edges that are within the buffered bounding box
# 1) Nodes
G_pt = nx.DiGraph()
node_dict = pt_graph_clip.to_dict(orient='index')
G_pt.add_nodes_from(node_dict.keys())
nx.set_node_attributes(G_pt, node_dict)
# 2) Edges
df_pt_edges = nx.to_pandas_edgelist(G_pt_full)
df_edges_keep = df_pt_edges.loc[(df_pt_edges['source'].isin(pt_graph_clip.index.tolist())) & (df_pt_edges['target'].isin(pt_graph_clip.index.tolist()))]
df_edges_keep.set_index(['source','target'], inplace=True)
edge_dict = df_edges_keep.to_dict(orient='index')
G_pt.add_edges_from(edge_dict.keys())
nx.set_edge_attributes(G_pt, edge_dict)
# plot for visualization
# node_color = ['black' if n.startswith('ps') else 'blue' for n in G_pt.nodes]
# edge_color = ['grey' if (e[0].startswith('ps') and e[1].startswith('rt')) | (e[0].startswith('rt') and e[1].startswith('ps')) else 'black' for e in G_pt.edges]
# ax = ut.draw_graph(G_pt, node_color, {'physical stop':'black', 'route node':'blue'}, edge_color, 'solid')
# ax.set_title('Public Transit Network')

#%% SCOOTER graph:
    # Attributes: TT, reliability ,risk,
G_sc = G_bike.copy()
#TODO: filter out streets with speed limit > 25
G_sc = ut.rename_nodes(G_sc, 'sc')
nx.set_node_attributes(G_sc, 'sc', 'nwk_type')
nx.set_edge_attributes(G_sc, 'sc', 'mode_type')
nx.set_node_attributes(G_sc, 'sc', 'node_type')# all nodes have same node type (i.e. no special nodes)

# # price and discomf are time-dependent
# for e in G_sc.edges:
#     #for i in range(num_intervals):
#     price = (conf.config_data['Price_Params']['scoot']['ppmin']/60) * G_sc.edges[e]['0_avg_TT_sec']  # op cost per edge
#     G_sc.edges[e]['0_price'] = price
#     #discomf = conf.config_data['Discomfort_Params']['scoot'] # * G_sc.edges[e]['interval'+str(i)+'_avg_TT_min']
#     G_sc.edges[e]['0_discomfort'] = conf.config_data['Discomfort_Params']['scoot']

#%% ZIPCAR (or generally, carshare) graph
# read data which was obtained from Google MyMaps
filepath = os.path.join(cwd,'Data','Input_Data','Zipcar_Depot.csv')
df_zip = pd.read_csv(filepath)
gdf_zip = gpd.GeoDataFrame(data=df_zip, geometry=df_zip['WKT'].apply(wkt.loads), crs='EPSG:4326').reset_index()[['index','geometry']]
gdf_zip['pos'] = tuple(zip(gdf_zip.geometry.x, gdf_zip.geometry.y)) # add position
gdf_zip.rename(columns={'index':'id'}, inplace=True)
gdf_zip_clip = gpd.clip(gdf_zip, study_area_gdf)

# steps: copy the driving graph. add parking cnx edges. add zip depot cnx edges
G_z = G_drive.copy()
G_z = ut.rename_nodes(G_z, 'z')
nx.set_node_attributes(G_z, 'z', 'nwk_type')
nx.set_node_attributes(G_z, 'z', 'node_type')
nx.set_edge_attributes(G_z, 'z', 'mode_type')

# add price and discomf attributes, which are time-dep 
# for e in G_z.edges:
#     #for i in range(num_intervals):
#     price = (conf.config_data['Price_Params']['zip']['ppmin']/60) * G_z.edges[e]['0_avg_TT_sec']  # op cost per edge
#     G_z.edges[e]['0_price'] = price
#     #discomf = conf.config_data['Discomfort_Params']['zip'] * G_z.edges[e]['interval'+str(i)+'_avg_TT_min']
#     G_z.edges[e]['0_discomfort'] = conf.config_data['Discomfort_Params']['zip']

# add parking cnx edges
G_z = ut.add_depots_cnx_edges(gdf_parking_nodes, gdf_drive_nodes, 'kz', 'z', 'zip', num_intervals, G_z, 'to_depot')
# add depot cnx edges
G_z = ut.add_depots_cnx_edges(gdf_zip_clip, gdf_drive_nodes, 'zd', 'z', 'zip', num_intervals, G_z, 'from_depot')

# rename mode_type of parking edges
for e in G_z.edges:
    if e[1].startswith('k'):
        G_z.edges[e]['mode_type'] = 'park'

# # add cost of parking to zipcar: include parking rate + rate associated with zipcar rental
# park_hours = conf.config_data['Supernetwork']['num_park_hours']
# for e in G_z.edges:
#     if e[1].startswith('kz'):  # if an edge leading into a parking node
#         parking_rate = G_z.nodes[e[1]]['float_rate']  # rate per hour
#         zip_rate = conf.config_data['Price_Params']['zip']['ppmin']*60 # zip rate per hour
#         G_z.edges[e]['0_price'] = park_hours * (parking_rate + zip_rate)
        
# plot for visualization
node_color = ['blue' if n.startswith('zd') else 'red' if n.startswith('k') else 'black' for n in G_z.nodes]
edge_color = ['grey' if e[0].startswith('z') and e[1].startswith('z') else 'magenta' for e in G_z.edges]
ax = ut.draw_graph(G_z, node_color, {'road intersection':'black', 'depot':'blue', 'park':'red'}, edge_color, 'solid')
ax.set_title('Personal Vehicle Network')
