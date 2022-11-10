#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 17:52:13 2022

@author: lindsaygraff
"""

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

# conf.study_area_gdf = gpd.read_file(os.path.join(filepath, 'pgh_study_area.csv'))
# read graphs that were created in 'process_street_centerlines.py'
cwd = os.getcwd()
with open(os.path.join(cwd, 'Data', 'Output_Data', 'G_drive.pkl'), 'rb') as inp:
    G_drive = pickle.load(inp)
with open(os.path.join(cwd, 'Data', 'Output_Data', 'G_bike.pkl'), 'rb') as inp:
    G_bike = pickle.load(inp)

#%%
# Here we build the travel time multiplier as a function of time 
# some arbitary piecewise function
# add travel time by interval
len_period = int(conf.config_data['Time_Intervals']['len_period'])
num_intervals = int(conf.config_data['Time_Intervals']['len_period'] / conf.config_data['Time_Intervals']['interval_spacing']) + 1

x = np.linspace(0, len_period, num_intervals )  # x is time [min past] relative to 07:00 AM
y = np.piecewise(x, [x < 15, ((x>=15) & (x<45)), ((x>=45) & (x<=75)), ((x>75) & (x<105)), x >= 105],
                 [1, lambda x: (0.5/30)*(x - 30) + 1.25, 1.5, lambda x: (-0.5/30)*(x - 90) + 1.25 , 1])
plt.plot(x, y, 'o', color='black', zorder=2);
plt.plot(x, y, color='red', zorder=1);
plt.xlabel('Time (minutes relative to 07:00AM)')
plt.ylabel('Travel time multiplier \n (relative to baseline)')

#%% add avg travel time to G_drive and G_bike
for i in range(num_intervals):
    
    # DRIVE
    for e in G_drive.edges:
        # travel time: avg_TT = TT_multiplier * (distance / speed_limit)
        G_drive.edges[e]['interval' + str(i) + '_avg_TT_min'] =  (G_drive.edges[e]['length_m'] / 
                                                                  conf.config_data['Conversion_Factors']['meters_in_mile'] /
                                                                  G_drive.edges[e]['speed_lim'] * 60 * y[i])
        # reliability
        # (maybe: also evaluate road type i.e. residential roads may not have high reliability mult)
        G_drive.edges[e]['interval' + str(i) + '_reliability'] = conf.config_data['Reliability_Params']['drive'] * G_drive.edges[e]['interval' + str(i) + '_avg_TT_min']
        # risk: remove TT dependence
        G_drive.edges[e]['interval' + str(i) + '_risk'] = G_drive.edges[e]['risk_idx_drive'] #* G_drive.edges[e]['interval' + str(i) + '_avg_TT_min']
    
    # BIKE
    for e in G_bike.edges:
        # travel time: avg_TT = TT_multiplier * (distance / speed_limit)
        G_bike.edges[e]['interval' + str(i) + '_avg_TT_min'] =  (G_bike.edges[e]['length_m'] / 
                                                                  conf.config_data['Speed_Params']['bike'] / 60)
        # reliability
        # (maybe: also evaluate road type i.e. residential roads may not have high reliability mult)
        G_bike.edges[e]['interval' + str(i) + '_reliability'] = conf.config_data['Reliability_Params']['bike'] * G_bike.edges[e]['interval' + str(i) + '_avg_TT_min']
        # risk: remove TT dependence
        G_bike.edges[e]['interval' + str(i) + '_risk'] = G_bike.edges[e]['risk_idx_bike'] #* G_bike.edges[e]['interval' + str(i) + '_avg_TT_min']

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

#%%
# TNC graph: 
    # Attributes: TT, reliability, risk, price, discomfort
G_tnc = G_drive.copy()
nx.set_node_attributes(G_tnc, 't', 'nwk_type')  
nx.set_node_attributes(G_tnc, 't', 'node_type')# all nodes have same node type (i.e. no special nodes)
nx.set_edge_attributes(G_tnc, 't', 'mode_type')
G_tnc = ut.rename_nodes(G_tnc, 't')

# add TNC price
TNC_ppmile = conf.config_data['Price_Params']['TNC']['ppmile']
TNC_ppmin = conf.config_data['Price_Params']['TNC']['ppmin']
miles_in_km = conf.config_data['Conversion_Factors']['miles_in_km']
for e in G_tnc.edges:
    for i in range(num_intervals):
        G_tnc.edges[e]['interval'+str(i)+'_price'] = (TNC_ppmin * G_tnc.edges[e]['interval'+str(i)+'_avg_TT_min'] +
                                                      TNC_ppmile * miles_in_km * G_tnc.edges[e]['length_m']/1000)
        G_tnc.edges[e]['interval'+str(i)+'_discomfort'] = conf.config_data['Discomfort_Params']['TNC']
# the five attributes have been accounted for: avg_TT, reliability ,risk, price, reliability 

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
meters_in_mile = conf.config_data['Conversion_Factors']['meters_in_mile']
for e in G_pv.edges:
    price = pv_ppmile = conf.config_data['Price_Params']['pv']['ppmile'] * (G_pv.edges[e]['length_m'] / meters_in_mile)  # op cost per edge
    #G_pv.edges[e]['price'] = price
    price_attr = dict(zip(['interval'+str(i)+'_price' for i in range(num_intervals)], num_intervals * [price]))
    nx.set_edge_attributes(G_pv, {e: price_attr})
    for i in range(num_intervals):
        G_pv.edges[e]['interval'+str(i)+'_discomfort'] = conf.config_data['Discomfort_Params']['pv']

# join parking nodes and connection edges to the personal vehicle network
filepath = os.path.join(cwd, 'Data', 'Output_Data')
gdf_parking_nodes = gpd.read_file(os.path.join(filepath, 'parking_points.csv'))
#gdf_parking_nodes_clip = gpd.clip(gdf_parking_nodes, conf.study_area_gdf).reset_index().drop(columns='index')
gdf_parking_nodes['pos'] = tuple(zip(gdf_parking_nodes['longitude'], gdf_parking_nodes['latitude']))  # add position
gdf_parking_nodes.insert(0, 'id', gdf_parking_nodes.index)  # add ID to each parking node

G_pv = ut.add_depots_cnx_edges(gdf_parking_nodes, gdf_drive_nodes, # ['ID','pos','zone','float_rate'],
                               'k', 'pv', 'pv', num_intervals, G_pv, 'to_depot')

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

for e in G_pb.edges:
    avg_TT_min =  G_pb.edges[e]['length_m'] / conf.config_data['Speed_Params']['bike'] / 60
    price = conf.config_data['Price_Params']['pb']['ppmin'] * avg_TT_min  # op cost per edge (which is 0)
    price_attr = dict(zip(['interval'+str(i)+'_price' for i in range(num_intervals)], num_intervals * [price]))
    nx.set_edge_attributes(G_pb, {e: price_attr})

    discomf = conf.config_data['Discomfort_Params']['pb'] #* avg_TT_min
    discomf_attr = dict(zip(['interval'+str(i)+'_discomfort' for i in range(num_intervals)], 
                             num_intervals * [discomf]))
    nx.set_edge_attributes(G_pb, {e: discomf_attr})


#%% BIKESHARE graph:
    # Attributes: TT, reliability, risk, price, discomfort
    # *except* connection edges do not yet have all 5
bs_filepath = os.path.join(cwd, 'Data', 'Input_Data', 'pgh_bikeshare_depot_q3_2021.csv')
G_bs = build_bikeshare_graph(G_bike, bs_filepath, 'Latitude', 'Longitude', 
                             'Station #', 'Station Name', '# of Racks', num_intervals, gdf_bike_nodes)

#plot for visualization
node_color = ['black' if n.startswith('bsd') else 'blue' for n in G_bs.nodes]
edge_color = ['grey' if e[0].startswith('bs') and e[1].startswith('bs') else 'magenta' for e in G_bs.edges]
ax = ut.draw_graph(G_bs, node_color, {'road intersection':'black', 'pnr':'blue'}, edge_color, 'solid')
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

#%% Reduce the size of the PT network through a bounding box approach:
    # Find the bounding box of the pgh_study_area polygon. Extend this bounding box by x miles. Then clip the PT network by this extended bounding box

df = pd.DataFrame.from_dict(dict(G_pt_full.nodes), orient="index").reset_index()
#gdf_pt = gpd.GeoDataFrame(data=df, geometry=df.pos)
df[['x','y']] = pd.DataFrame(df.pos.tolist())
gdf_ptnodes = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x,df.y))
gdf_ptnodes.head(3)
#bbox_study_area = conf.study_area_gdf['geometry'].bounds.T.to_dict()[0]  # bounding box of neighborhood polygon layer
bbox_df = study_area_gdf['geometry'].bounds
x = 0.5
# extend the bounds of the study area
bbox_df['newminx'] = bbox_df['minx'] - 1/69 * x  # 1 degree/69 mile
bbox_df['newmaxx'] = bbox_df['maxx'] + 1/69 * x
bbox_df['newminy'] = bbox_df['miny'] - 1/69 * x 
bbox_df['newmaxy'] = bbox_df['maxy'] + 1/69 * x
# define new points that will be used to define an extended bounding box
pt1 = geometry.Point(bbox_df.newminx, bbox_df.newminy)
pt2 = geometry.Point(bbox_df.newminx, bbox_df.newmaxy)
pt3 = geometry.Point(bbox_df.newmaxx, bbox_df.newmaxy)
pt4 = geometry.Point(bbox_df.newmaxx, bbox_df.newminy)
bbox_new = geometry.Polygon((pt1,pt2,pt3,pt4))
bbox_new_gdf = gpd.GeoDataFrame(gpd.GeoSeries(bbox_new), columns=['geometry'])
# check that this worked
fig,ax = plt.subplots(figsize=(4,4))
study_area_gdf.plot(ax=ax, color='blue')
bbox_new_gdf.plot(ax=ax, color='green', alpha=.4)
# clip the list of all pt nodes to just those within the new bbox
pt_graph_clip = gpd.clip(gdf_ptnodes, bbox_new_gdf)
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
G_sc = ut.rename_nodes(G_sc, 'sc')
nx.set_node_attributes(G_sc, 'sc', 'nwk_type')
nx.set_edge_attributes(G_sc, 'sc', 'mode_type')
nx.set_node_attributes(G_sc, 'sc', 'node_type')# all nodes have same node type (i.e. no special nodes)

# price and discomf are time-dependent
for e in G_sc.edges:
    for i in range(num_intervals):
        price = conf.config_data['Price_Params']['scoot']['ppmin'] * G_sc.edges[e]['interval'+str(i)+'_avg_TT_min']  # op cost per edge
        G_sc.edges[e]['interval'+str(i)+'_price'] = price
        #discomf = conf.config_data['Discomfort_Params']['scoot'] # * G_sc.edges[e]['interval'+str(i)+'_avg_TT_min']
        G_sc.edges[e]['interval'+str(i)+'_discomfort'] = conf.config_data['Discomfort_Params']['scoot']

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
for e in G_z.edges:
    for i in range(num_intervals):
        price = conf.config_data['Price_Params']['zip']['ppmin'] * G_z.edges[e]['interval'+str(i)+'_avg_TT_min']  # op cost per edge
        G_z.edges[e]['interval'+str(i)+'_price'] = price
        #discomf = conf.config_data['Discomfort_Params']['zip'] * G_z.edges[e]['interval'+str(i)+'_avg_TT_min']
        G_z.edges[e]['interval'+str(i)+'_discomfort'] = conf.config_data['Discomfort_Params']['zip']

# add parking cnx edges
G_z = ut.add_depots_cnx_edges(gdf_parking_nodes, gdf_drive_nodes, # ['ID','pos','zone','float_rate'],
                               'kz', 'z', 'zip', num_intervals, G_z, 'to_depot')
# add depot cnx edges
G_z = ut.add_depots_cnx_edges(gdf_zip_clip, gdf_drive_nodes, #['ID','pos'],
                               'zd', 'z', 'zip', num_intervals, G_z, 'from_depot')


# plot for visualization
node_color = ['blue' if n.startswith('zd') else 'red' if n.startswith('k') else 'black' for n in G_z.nodes]
edge_color = ['grey' if e[0].startswith('z') and e[1].startswith('z') else 'magenta' for e in G_z.edges]
ax = ut.draw_graph(G_z, node_color, {'road intersection':'black', 'depot':'blue', 'park':'red'}, edge_color, 'solid')
ax.set_title('Personal Vehicle Network')


#%% successful testing
for e in G_z.edges:
    if e[1].startswith('zd'):
        print(e)

#%%

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
# gdf_bs_clip = gpd.clip(gdf_bs, conf.study_area_gdf).reset_index().drop(columns=['index']).rename(columns={'Station #': 'ID'})

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
# gdf_parking_nodes_clip = gpd.clip(gdf_parking_nodes, conf.study_area_gdf).reset_index().drop(columns='index')
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
