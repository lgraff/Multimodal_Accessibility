#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 11:41:27 2022

@author: lindsaygraff
"""

# import libraries
import geopandas as gpd
import pandas as pd
import os
import networkx as nx 
import numpy as np
import matplotlib.pyplot as plt
from shapely import wkt
#import util_functions as ut
from create_study_area import study_area_gdf
import util_functions as ut
#import global_ as gl
#from global_ import config_data, study_area_gdf
from build_unimodal_graphs import G_tnc, G_pv, G_pb, G_bs, G_pt, G_sc, G_z

# Establish parameters for which 
# ******for testing, remember to put into config_data****
TNC_wait_time = 7  # [min]

# this dict defines which graphs correspond to each mode type 
all_graphs_dict = {'t':G_tnc, 'pv':G_pv, 'pb':G_pb, 'bs':G_bs, 'pt':G_pt, 'sc':G_sc, 'z':G_z}

# this dict defines the node names corresponding to each mode type 
all_modes_nodes = {'bs':['bs'], 'pt':['ps','rt'], 't':['t'], 'sc':['sc'], 
                   'pv':['pv','k'], 'pb':['pb'], 'z':['zd','z','kz']}

# define which nodes are fixed and which come from flexible networks 
all_fix_pre = ['bs','ps','k', 'zd', 'kz']  # prefix for fixed nodes
all_flex_pre = ['t', 'pb', 'pv', 'sc']  # prefix for flexible dropoff nodes

modes_included = ut.config_data['Supernetwork']['modes_included']

# this dict defines which modes and nodes are included in the supernetwork
modes_nodes_included = {k:v for k,v in all_modes_nodes.items() if k in modes_included}
networks_included = [all_graphs_dict[m] for m in modes_included]  # set([all_graphs_dict[m] for m in modes_included])

# prefixes for fixed nodes in the supernetwork
fix_pre = [n for nodes in modes_nodes_included.values() for n in nodes if n in all_fix_pre]
# prefixes for flex nodes in the supernetwork
flex_pre = [n for nodes in modes_nodes_included.values() for n in nodes if n in all_flex_pre]

#%%
# combine unimodal networks 
G_u = nx.union_all(networks_included)

#%%
nid_map, coord_matrix = ut.get_coord_matrix(G_u)
# coord matrix is np array where row i contains the lat/long pair of node i
# node ID map is of form {id_number: node_name}. also make inverse node ID map of form {node_name: id_number}

inv_nid_map = dict(zip(nid_map.values(), nid_map.keys()))  
# separate nid map into fixed and flex
nid_map_fixed = {key:val for key,val in nid_map.items() if ut.mode(val) in fix_pre}  # nid_map for fixed network nodes
nid_map_flex = {key:val for key,val in nid_map.items() if ut.mode(val) in flex_pre}  # nid_map for flex network nodes













