#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 15:20:10 2022

@author: lindsaygraff
"""

# import libraries
import yaml
import geopandas as gpd
import shapely
from shapely.geometry import box, LineString, Point,MultiPoint
from shapely.ops import nearest_points
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import networkx as nx
import re
from math import radians, degrees, sin, cos, asin, acos, sqrt, floor
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from shapely.geometry import shape
import fiona
import itertools
import osmnx as ox
import ckanapi

# load config file
def load_config(config_filename):
    with open(config_filename, "r") as yamlfile:
        config_data = yaml.load(yamlfile, Loader=yaml.FullLoader)
        print("Read successful")
    return config_data

config_data = load_config('config.yaml')

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

def calc_bike_risk_index(row):
    if((row['highway'] == 'cycleway') | (row['bikeway_type'] in (['Bike Lanes','Protected Bike Lane']))):
        risk_idx = 1
    elif(row['highway'] in (['motorway','trunk','trunk_link','primary','primary_link'])):
        risk_idx = 100000
    else:
        risk_idx = config_data['Risk_Parameters']['risk_weight_active']   # this is a parameter to be adjusted. idea is that non-bikelane road is 25% more dangerous
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
from sklearn.neighbors import BallTree

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
                         ref_id_prefix, movement_speed, time_interval_cols, G_ref, is_twoway_cnx=False):
    # inputs: 
    # output: 
        
    # get point in reference network nearest to each parking node; only keep the ID of the point of the length 
    # of the segment connecting the parking node to its nearest neighbor
    nn = nearest_neighbor(gdf_depot_nodes, gdf_ref_nodes, 'y', 'x', return_dist=True)[['nn_osmid', 'length']]
    #cols_keep = ['ID', 'pos', 'zone', 'float_rate']
    cols_keep = depot_cols_keep + ['nn_osmid', 'length']
    gdf_depot_nodes = pd.concat([gdf_depot_nodes, nn], axis=1)[cols_keep]
    gdf_depot_nodes['ID'] = gdf_depot_nodes.apply(lambda row: depot_id_prefix + str(int(row['ID'])), axis=1)

    # build cnx edges
    gdf_cnx_edges = gdf_depot_nodes[['ID', 'nn_osmid', 'length']]
    #gdf_pv_parking_edges = gdf_parking_edges_clip.copy()
    gdf_cnx_edges['nn_osmid'] = gdf_cnx_edges.apply(lambda row: ref_id_prefix + str(int(row['nn_osmid'])), axis=1)
    cnx_attr = (gdf_cnx_edges['length'] / movement_speed / 60).rename('avg_TT_min')  # m / (m/s) / (60s/min)
    cnx_attr = pd.concat([cnx_attr] * len(time_interval_cols), axis=1)
    cnx_attr.columns = time_interval_cols
    gdf_cnx_edges = pd.concat([gdf_cnx_edges, cnx_attr], axis=1)
    gdf_cnx_edges.set_index(['nn_osmid', 'ID'], inplace=True)
    cnx_edge_dict = gdf_cnx_edges.to_dict(orient='index')

    # add connection edges to the graph. then add nodes and their attributes (position, zone name, rate)
    G_ref.add_edges_from(list(cnx_edge_dict.keys()))
    nx.set_edge_attributes(G_ref, cnx_edge_dict)
    gdf_depot_nodes.set_index(['ID'], inplace=True)
    node_dict = gdf_depot_nodes.drop(columns=['nn_osmid', 'length']).to_dict(orient='index')
    nx.set_node_attributes(G_ref, node_dict)   
    
    
    if is_twoway_cnx:
        #print('************88')
        oneway_edges = list(zip(*list(cnx_edge_dict.keys())))
        other_edges = list(zip(oneway_edges[1], oneway_edges[0]))
       # print(oneway_edges)
        #print('------------')
        #print(other_edges)
        other_edges_attr = dict(zip(other_edges, cnx_edge_dict.values()))
        G_ref.add_edges_from(list(other_edges))
        nx.set_edge_attributes(G_ref, other_edges_attr)
        
    return G_ref
    
    
    
    
    