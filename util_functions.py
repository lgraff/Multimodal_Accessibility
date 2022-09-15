#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 15:20:10 2022

@author: lindsaygraff
"""

# import libraries
from sklearn.neighbors import BallTree
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import re
import ckanapi
import config as conf
#import yaml
#import geopandas as gpd
#import os
#from global_ import conf.config_data, study_area_gdf


# # load conf.config file
# def load_config(config_filename):
#     with open(config_filename, "r") as yamlfile:
#         conf.config_data = yaml.load(yamlfile, Loader=yaml.FullLoader)
#         print("Read successful")
#     return conf.config_data

# conf.config_data = load_config('conf.config.yaml')

# import global variables conf.config_data (a dict of dicts) and study_area_gdf (polygon layer)
#from global_ import conf.config_data, study_area_gdf  

# def load_conf.config(conf.config_filename):
#     with open(conf.config_filename, "r") as yamlfile:
#         conf.config_data = yaml.load(yamlfile, Loader=yaml.FullLoader)
#         print("Read successful")
#     return conf.config_data

#conf.config_data = load_conf.config('conf.config.yaml')
#def get_study_area_gdf(study_area_filename):

def get_resource_data(site,resource_id,count=50):
    # Use the datastore_search API endpoint to get <count> records from
    # a CKAN resource.
    ckan = ckanapi.RemoteCKAN(site)
    response = ckan.action.datastore_search(id=resource_id, limit=count)
    data = response['records']
    return data

# convert lat and long in deg-min-sec to decimal degrees
def dms2dd(angle):
    #s = "40 27:10.501"
    dec, ms = tuple(angle.split(' '))
    m, s = tuple(ms.split(':'))
    dd = float(dec) + float(m)/60 + float(s)/3600
    return dd

def flatten_proj(gdfs, crs):
    for g in gdfs:
        g.to_crs(crs=crs, inplace=True)
        
# convert gdfs to networkx graph
def gdfs_to_nxgraph(gdf_edges, gdf_nodes, source_col, target_col, nodeid_col, lat_col, long_col, edge_attr):
    G = nx.from_pandas_edgelist(gdf_edges, source=source_col, target=target_col,
                                edge_attr = edge_attr,
                                create_using=nx.DiGraph())
    gdf_nodes.index = gdf_nodes[nodeid_col]
    gdf_nodes['pos'] = tuple(zip(gdf_nodes[long_col], gdf_nodes[lat_col]))  # add position
    node_dict = gdf_nodes.drop(columns=[nodeid_col]).to_dict(orient='index')
    G.add_nodes_from(list(node_dict.keys()))
    nx.set_node_attributes(G, node_dict)
    return G

# def gdfnodes_to_nxgraph(gdf_nodes, id_col, lat_col, long_col):
#     gdf_nodes.index = gdf_nodes[id_col]
#     gdf_nodes['pos'] = tuple(zip(gdf_nodes[long_col], gdf_nodes[lat_col]))  # add position
#     node_dict = gdf_nodes.drop(columns=[id_col]).to_dict(orient='index')
#     G = nx.DiGraph()
#     G.add_nodes_from(list(node_dict.keys()))
#     nx.set_node_attributes(G, node_dict)
#     return G    

def calc_bike_risk_index(row, risk_weight_active):
    if((row['highway'] == 'cycleway') | (row['bikeway_type'] in (['Bike Lanes','Protected Bike Lane']))):
        risk_idx = 1
    elif(row['highway'] in (['motorway','trunk','trunk_link','primary','primary_link'])):
        risk_idx = 100000
    else:
        # this is a parameter to be adjusted. idea is that non-bikelane road is 25% more dangerous
        risk_idx = risk_weight_active 
    return risk_idx

def draw_graph(G, node_color, node_cmap, edge_color):
    # draw the graph in networkx
    node_coords = nx.get_node_attributes(G, 'pos')    
    fig, ax = plt.subplots(figsize=(20,20))
    nx.draw(G, pos=node_coords, with_labels=False, font_color='white',  font_weight = 'bold',
            node_size=20, node_color=node_color, edge_color=edge_color, arrowsize=10, ax=ax)
    # add legend for node color    
    inv_node_cmap = dict(zip(node_cmap.values(), node_cmap.keys()))
    for v in set(inv_node_cmap.keys()):
        ax.scatter([],[], c=v, label=inv_node_cmap[v])
    ax.legend(loc = 'upper right')
    return ax

# https://autogis-site.readthedocs.io/en/latest/notebooks/L3/06_nearest-neighbor-faster.html


def get_nearest(src_points, candidates, k_neighbors=1):
    """Find nearest neighbors for all source points from a set of candidate points"""

    # Create tree from the candidate points
    tree = BallTree(candidates, leaf_size=15, metric='haversine')

    # Find closest points and distances
    distances, indices = tree.query(src_points, k=k_neighbors)

    # Transpose to get distances and indices into arrays
    distances = distances.transpose()
    indices = indices.transpose()

    # Get closest indices and distances (i.e. array at index 0)
    # note: for the second closest points, you would take index 1, etc.
    closest = indices[0]
    closest_dist = distances[0]

    # Return indices and distances
    return (closest, closest_dist)


def nearest_neighbor(left_gdf, right_gdf, right_lat_col, right_lon_col, return_dist=False):
    """
    For each point in left_gdf, find closest point in right GeoDataFrame and return them.
    
    NOTICE: Assumes that the input Points are in WGS84 projection (lat/lon).
    """
    
    left_geom_col = left_gdf.geometry.name
    right_geom_col = right_gdf.geometry.name
    
    # Ensure that index in right gdf is formed of sequential numbers
    right = right_gdf.copy().reset_index(drop=True)
    
    # Parse coordinates from points and insert them into a numpy array as RADIANS
    # Notice: should be in Lat/Lon format 
    left_radians = np.array(left_gdf[left_geom_col].apply(lambda geom: (geom.y * np.pi / 180, geom.x * np.pi / 180)).to_list())
    right_radians = np.array(right[right_geom_col].apply(lambda geom: (geom.y * np.pi / 180, geom.x * np.pi / 180)).to_list())
    
    # Find the nearest points
    # -----------------------
    # closest ==> index in right_gdf that corresponds to the closest point
    # dist ==> distance between the nearest neighbors (in meters)
    
    closest, dist = get_nearest(src_points=left_radians, candidates=right_radians)
    
    
    # Return points from right GeoDataFrame that are closest to points in left GeoDataFrame
    closest_points = right.loc[closest]
    
    # Ensure that the index corresponds the one in left_gdf
    closest_points = closest_points.reset_index(drop=True)
    
    # Add distance if requested 
    if return_dist:
        # Convert to meters from radians
        earth_radius = 6371000  # meters
        closest_points['length'] = dist * earth_radius
    
    #return closest_points[cols_keep].rename(columns = {'osmid': 'nn_osmid'})
    return closest_points.drop(columns=[right_geom_col, right_lat_col, right_lon_col]).rename(columns = {'osmid': 'nn_osmid'})

def rename_nodes(G, prefix):
    new_nodename = [prefix + re.sub('\D', '', str(i)) for i in G.nodes]
    namemap = dict(zip(G.nodes, new_nodename))
    G = nx.relabel_nodes(G, namemap, True)
    return G

def add_depots_cnx_edges(gdf_depot_nodes, gdf_ref_nodes, depot_cols_keep, depot_id_prefix, 
                         ref_id_prefix, cnx_edge_movement_type, num_time_intervals, G_ref, 
                         cnx_direction):
    # inputs: 
    # output: 
        
    # get point in reference network nearest to each parking node; only keep the ID of the point of the length 
    # of the segment connecting the parking node to its nearest neighbor
    nn = nearest_neighbor(gdf_depot_nodes, gdf_ref_nodes, 'y', 'x', return_dist=True)[['nn_osmid', 'length']].reset_index(drop=True)
    #cols_keep = ['ID', 'pos', 'zone', 'float_rate']
    cols_keep = depot_cols_keep + ['nn_osmid', 'length']
    gdf_depot_nodes = pd.concat([gdf_depot_nodes.reset_index(drop=True), nn], axis=1)[cols_keep]
    gdf_depot_nodes['ID'] = gdf_depot_nodes.apply(lambda row: depot_id_prefix + str(int(row['ID'])), axis=1)

    # build cnx edges
    gdf_cnx_edges = gdf_depot_nodes[['ID', 'nn_osmid', 'length']]
    #gdf_pv_parking_edges = gdf_parking_edges_clip.copy()
    gdf_cnx_edges.loc[:,'nn_osmid'] = ref_id_prefix + gdf_cnx_edges.loc[:,'nn_osmid'].astype('int32').astype('str')
    #gdf_cnx_edges.loc[:,'nn_osmid'] = gdf_cnx_edges.loc[:, 'nn_osmid'].apply(lambda x: ref_id_prefix + str(int(x))) #, axis=1)
    # add cnx edge travel time for each time interval
    movement_speed = conf.config_data['Connection_Edge_Speed'][cnx_edge_movement_type]
    cnx_attr = (gdf_cnx_edges['length'] / movement_speed / 60).rename('avg_TT_min')  # m / (m/s) / (60s/min)
    cnx_attr = pd.concat([cnx_attr] * num_time_intervals, axis=1)
    cnx_attr.columns = (['interval'+str(i) + '_' + 'avg_TT_min' for i in range(num_time_intervals)])
    
    # these cnx edges go FROM ref network TO depot network
    gdf_cnx_edges = pd.concat([gdf_cnx_edges, cnx_attr], axis=1)
    gdf_cnx_edges.set_index(['nn_osmid', 'ID'], inplace=True)  # FROM ref network TO depot network
    cnx_edge_dict = gdf_cnx_edges.to_dict(orient='index')
    to_depot_edges = list(zip(*list(cnx_edge_dict.keys())))
    
    #print(to_depot_edges)

    # also create edges FROM depot network TO ref network
    from_depot_edges = list(zip(to_depot_edges[1], to_depot_edges[0]))
    from_depot_edges_attr = dict(zip(from_depot_edges, cnx_edge_dict.values()))
    
    #print(from_depot_edges)
    # for node attributes
    gdf_depot_nodes.set_index(['ID'], inplace=True)
    node_dict = gdf_depot_nodes.drop(columns=['nn_osmid', 'length']).to_dict(orient='index')
    
    # add edges based on user-specified direction
    if cnx_direction == 'to_depot':
        # add connection edges to the graph. then add nodes and their attributes (depot_cols_keep)
        G_ref.add_edges_from(list(cnx_edge_dict.keys()))
        nx.set_edge_attributes(G_ref, cnx_edge_dict)
        nx.set_node_attributes(G_ref, node_dict)   
        # also add attributes for reliability, risk, price, and discomfort   
    elif cnx_direction == 'from_depot':
        G_ref.add_edges_from(list(from_depot_edges))
        nx.set_edge_attributes(G_ref, from_depot_edges_attr)
        nx.set_node_attributes(G_ref, node_dict)
    elif cnx_direction == 'both':
        G_ref.add_edges_from(list(cnx_edge_dict.keys()))  # one way
        nx.set_edge_attributes(G_ref, cnx_edge_dict)
        G_ref.add_edges_from(list(from_depot_edges))  # other way
        nx.set_edge_attributes(G_ref, from_depot_edges_attr)
        nx.set_node_attributes(G_ref, node_dict)

    # add reliability, risk, price, and discomfort    
    # assumptions: 95% TT is the same as avg TT; 
    # ** TO DO below ** 

    all_cnx_edge_list = [e for e in G_ref.edges if (
        (e[0].startswith(depot_id_prefix) & e[1].startswith(ref_id_prefix)) | 
        (e[0].startswith(ref_id_prefix) & e[1].startswith(depot_id_prefix)))]
    
    # print(all_cnx_edge_list)
    # print('******')
    # print(conf.config_data)
    # print('******')
    
    # obtain proper price parameter
    if ref_id_prefix == 'z':
        price_param = conf.config_data['Price_Params']['zip']['ppmin']
    elif ref_id_prefix == 'pv':
        price_param = conf.config_data['Price_Params']['pv']['ppmile']
    elif ref_id_prefix == 'bs':
        price_param = conf.config_data['Price_Params']['bs']['ppmin']
          
    for e in all_cnx_edge_list:
        for i in range(num_time_intervals): 
            if ref_id_prefix == 'pv':                
                G_ref.edges[e]['interval' + str(i) + '_price'] = price_param * (
                    G_ref.edges[e]['length'] / conf.config_data['Conversion_Factors']['meters_in_mile'])  # price/mile * miles
            else:
                G_ref.edges[e]['interval' + str(i) + '_price'] = price_param * (
                    G_ref.edges[e]['interval' + str(i) + '_avg_TT_min'] )
            G_ref.edges[e]['interval' + str(i) + '_reliability'] = (conf.config_data['Reliability_Params'][cnx_edge_movement_type] *
                                                                    G_ref.edges[e]['interval' + str(i) + '_avg_TT_min'])
            G_ref.edges[e]['interval' + str(i) + '_risk'] =  G_ref.edges[e]['interval' + str(i) + '_avg_TT_min']
            G_ref.edges[e]['interval' + str(i) + '_discomfort'] = (conf.config_data['Discomfort_Params'][cnx_edge_movement_type] * 
                                                                   G_ref.edges[e]['interval' + str(i) + '_avg_TT_min'])
    
    return G_ref

def get_coord_matrix(G):
    coords_dict = nx.get_node_attributes(G, 'pos')
    nid_map = dict(zip(range(len(coords_dict.keys())), list(coords_dict.keys())))
    coord_matrix = np.array(list(coords_dict.values()))
    return (nid_map, coord_matrix)

# find the great circle distance between an input row (point) and a reference matrix (all other points)
# GCD: https://medium.com/@petehouston/calculate-distance-of-two-locations-on-earth-using-python-1501b1944d97#:~:text=The%20Great%20Circle%20distance%20formula,that%20the%20Earth%20is%20spherical.
# inputs: row (coords of single point), matrix_ref (coordinate matrix for all points)

# code source: https://github.com/gboeing/osmnx/blob/main/osmnx/distance.py
def calc_great_circle_dist(row, matrix_ref, earth_radius=6371009):
    y1 = np.deg2rad(row[1])  # y is latitude 
    y2 = np.deg2rad(matrix_ref[:,1])
    dy = y2 - y1

    x1 = np.deg2rad(row[0])
    x2 = np.deg2rad(matrix_ref[:,0])
    dx = x2 - x1

    h = np.sin(dy / 2) ** 2 + np.cos(y1) * np.cos(y2) * np.sin(dx / 2) ** 2
    h = np.minimum(1, h)  # protect against floating point errors
    arc = 2 * np.arcsin(np.sqrt(h))

    # return distance in units of earth_radius
    return arc * earth_radius

# returns the travel mode the corresponds to the node
def mode(node_name):
    mode_of_node = re.sub(r'[^a-zA-Z]', '', node_name)
    return mode_of_node

# returns walking catchment node for the node of interest
# inputs: nodeID of node interest, matrix of gc distances b/w all nodes, and max walking distance
# output: list of nodeIDs of all nodes within the wcz
def wcz(i, dist_matrix, W):
    catchment = np.where(dist_matrix[i] <= W)[0].tolist()
    if i in catchment:
        catchment.remove(i)  # remove self
    return catchment

# inputs: node of interest, matrix of gcd distances b/w all nodes, travel mode of interest, all nodes in the original graph (id+name)
# output: nodeID of the node in the component network of the travel mode of interest that is nearest to the input node of interest
def nn(i, dist_matrix, travel_mode, node_id_map):
    # subset the node_id_map for the nodes in the component network of the travel mode of interest
    nid_map_travel_mode = [key for key,val in node_id_map.items() if val.startswith(travel_mode)]   # this is a list of IDs
    # subset dist matrix for the nodes in the component network of the travel mode of interest
    dist_subset = dist_matrix[:, nid_map_travel_mode]
    # find the node in the component network of interest that is nearest to the input node of interest
    nn_dist = np.amin(dist_subset[i])
    nn_idx = np.argmin(dist_subset[i])
    # now map back to the original node ID
    original_nn_id = nid_map_travel_mode[nn_idx]
    original_nn_name = node_id_map[original_nn_id]
    return (original_nn_id, original_nn_name, nn_dist)

# for e in G_pb.edges:
#     price = conf.config_data['Price_Params']['pb_ppmin'] * G_pb.edges[e]['avg_TT_min']  # op cost per edge (which is 0)
#     price_attr = dict(zip(['interval'+str(i)+'_price' for i in range(num_intervals)], num_intervals * [price]))
#     nx.set_edge_attributes(G_pb, {e: price_attr})
    
    
    
    