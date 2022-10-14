#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 11:12:44 2022

@author: lindsaygraff
"""
# librarires
import os
import sys
import numpy as np
import pickle
import copy
import util_functions as ut
import networkx as nx
import config as conf
from supernetwork import Supernetwork
from buid_street_network import build_street_network
from create_study_area import create_study_area

# import importlib
# importlib.reload(sys.modules['od_connector'])
from od_connector import od_cnx

cwd = os.getcwd()

#%% 
#1) write new config file
#import write_config 
# 2) create the new study area:
    # create_study_area(os.path.join( os.getcwd(), 'Data', 'Input_Data', "Neighborhoods_.shp"), os.path.join(cwd, 'Data', 'Output_Data', 'study_area.csv') )
    # nhoodfpath = os.path.join( os.getcwd(), 'Data', 'Input_Data', "Neighborhoods_.shp")
    # outputfpath = os.path.join(cwd, 'Data', 'Output_Data', 'study_area.csv') 
# 3) build the street network
#build_street_network(conf.study_area_gdf, os.path.join(cwd, 'Data', 'Output_Data'))
# 4) add street safety info
# 5) build supernetwork (which includes as a subprocess building unimodal graphs)
#build_supernetwork(os.path.join(cwd, 'Data', 'Output_Data', 'G_super.pkl'))  # this saves as a .pkl file called G_super.pkl

# read pickled supernetwork (which includes unimodal networks joined by transfer edges)
cwd = os.getcwd()
with open(os.path.join(cwd, 'Data', 'Output_Data', 'G_super.pkl'), 'rb') as inp:
    G_super = pickle.load(inp)
print(G_super.fix_pre)
print(G_super.coord_matrix.shape)

#%% add on to the supernetwork: include org, dst, and od connectors 
G_super_od = copy.deepcopy(G_super)

# define od pair
o_coord = [-79.94868171046522, 40.416379503934145]
d_coord = [-79.92070090793109, 40.463543819430086]
# build od coord matrix
od_coords = np.array([o_coord, d_coord])
# add the org and dst nodes to the graph along with their positions 
G_super_od.graph.add_nodes_from([('org', {'pos': tuple(o_coord), 'nwk_type':'od', 'node_type':'od'}),  
                                 ('dst', {'pos': tuple(d_coord), 'nwk_type':'od', 'node_type':'od'})])

# 2 rows because 1 origin and 1 destination 
nrow, ncol = 2, len(G_super_od.nid_map) + 2  # add 2 to len(nid_map) to account for addition of org node and dst node

gcd_od = np.zeros([nrow, ncol])  
for j in range(nrow):  # first the org, then the dest
    dist_to_map_nodes = ut.calc_great_circle_dist(od_coords[j], G_super_od.coord_matrix) # dist from OD to modal graph nodes
    dist_to_od = ut.calc_great_circle_dist(od_coords[j], od_coords) # dist from O to O and O to D, then D to O and D to D
    gcd_od[j] = np.hstack((dist_to_map_nodes, dist_to_od)) # horizontally concatenate 
    
# now add gcd_dist_od to the original gcd_dist matrix
# result is gcd_dist_all, which contains the gcd dist b/w all pairs of nodes in the graph, including org and dst
gcd_dist_all = np.vstack((G_super_od.gcd_dist, gcd_od[:,:len(G_super_od.nid_map)]))
gcd_dist_all = np.hstack((gcd_dist_all, np.transpose(gcd_od)))

# adjust the nid_map and gcd_dist matrix to reflect the addition of org and dst
G_super_od.nid_map[max(G_super_od.nid_map.keys())+1] = 'org'
G_super_od.nid_map[max(G_super_od.nid_map.keys())+1] = 'dst'
G_super_od.gcd_dist = gcd_dist_all

# build od_cnx and add to graph
od_cnx_edges = od_cnx(G_super_od, o_coord, d_coord)
od_cnx_edges = [(e[0], e[1], od_cnx_edges[e])for e in od_cnx_edges.keys()] # convert od_cnx_edges to proper form so that they can be added to graph
G_super_od.graph.add_edges_from(od_cnx_edges)

# then save the object
cwd = os.getcwd()
ut.save_object(G_super_od, os.path.join(cwd, 'Data', 'Output_Data', 'G_super_od.pkl'))

#%%
# at this point, we have the supernetwork
# now convert to link cost file
df_link = nx.to_pandas_edgelist(G_super_od.graph)


#%% check for NA values
col_vals = ['mode_type', 'interval5_avg_TT_min', 'interval5_reliability', 'interval5_price', 'interval5_discomf', 'interval5_risk']
for c in col_vals:
    print(c)
    print(df_link[df_link[c].isna()][['source','target']])  
    print('~~~~~~~~~~')



# #%% **for visualization**: jitter the graphs and draw
# def jitter_nodes(G, network_type, jitter_param):
#     #G_adj = G.copy()
#     #print(G.nodes) #[node]['pos'])
#     # adjust the nodes positions in the copy 
#     nodes = [n for n in G.nodes.keys() if n not in ['org','dst'] if G.nodes[n]['nwk_type'] == network_type] 
#     for n in nodes:
#         adj_x = G.nodes[n]['pos'][0] + jitter_param
#         adj_y = G.nodes[n]['pos'][1] + jitter_param
#         nx.set_node_attributes(G, {n: {'pos_adj':(adj_x, adj_y)}})

# #%%
# adjusted_graph_dict = {}
# # adjust the graphs in the supernetwork using the jitter function
# # this is strictly for plotting
# # for j, m in enumerate(modes_included):
# #     print(j,m)
# #     #if m != 't':  # tnc graph is the "base" road network so it will not be adjusted
# #     G_adj = all_graphs_dict[m].copy()
# #     jitter_nodes(G_adj, jitter_param=(j/100)*2)
# #     adjusted_graph_dict[m] = G_adj

# modes_included = conf.config_data['Supernetwork']['modes_included']
# jitter_param_dict = {m: (j/100)*2 for j,m in enumerate(modes_included)}  # can adjust jitter param as necessary
# # also include adjusted pos for org/dst
# G_super_od.graph.nodes['org']['pos_adj'] = G_super_od.graph.nodes['org']['pos']
# G_super_od.graph.nodes['dst']['pos_adj'] = G_super_od.graph.nodes['dst']['pos']

# for m in modes_included:
#     jitter_nodes(G_super_od.graph, m, jitter_param_dict[m])
    
# # plot for visualization
# node_color_map = {'bs':'green', 'bsd':'lime', 'z':'blue', 'zd': 'skyblue', 'kz': 'dodgerblue',
#                   'sc':'darkviolet', 't':'red', 'ps':'brown', 'rt': 'orange' , 'od':'black'}
# node_color = [node_color_map[G_super_od.graph.nodes[n]['node_type']] for n in G_super_od.graph.nodes.keys()]
# edge_color = ['grey' if G_super_od.graph.edges[e]['mode_type'] == 'w' else 'gold' for e in G_super_od.graph.edges] 
# ax = ut.draw_graph(G_super_od.graph, node_color, node_color_map, edge_color, adjusted=True)
#%%
# # generate scooter data and establish distance to org and dist 
# num_days_of_data = conf.config_data['Scoot_Data_Generation']['num_days_of_data']
# num_obs = conf.config_data['Scoot_Data_Generation']['num_obs']
# num_intervals = int(conf.config_data['Time_Intervals']['len_period'] / conf.config_data['Time_Intervals']['interval_spacing']) + 1
# sc_costs_od = gen_data(G_super_od, num_days_of_data, num_intervals, num_obs, conf.bbox_study_area, od_cnx=True)
