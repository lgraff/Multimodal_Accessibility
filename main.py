#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 11:12:44 2022

@author: lindsaygraff
"""
# librarires
import os
import numpy as np
import pickle
import copy
import pandas as pd
import util_functions as ut
import networkx as nx
import config as conf
#from supernetwork import Supernetwork
#from build_unimodal_graphs import G_tnc, G_pv, G_pb, G_bs, G_pt, G_sc, G_z
from build_supernetwork import build_supernetwork
#from buid_street_network import build_street_network

# import importlib
# importlib.reload(sys.modules['od_connector'])
from od_connector import od_cnx

#%% build supernetwork, also save as pickled object for later use if necessary (avoid compiling it many times)
cwd = os.getcwd()
G_super = build_supernetwork(os.path.join(cwd, 'Data', 'Output_Data', 'G_super.pkl')) 
#%% 
#1) write new config file
#import write_config 
# 2) create the new study area:
    # create_study_area(os.path.join( os.getcwd(), 'Data', 'Input_Data', "Neighborhoods_.shp"), os.path.join(cwd, 'Data', 'Output_Data', 'study_area.csv') )

# 3) build the street network
#build_street_network(conf.study_area_gdf, os.path.join(cwd, 'Data', 'Output_Data'))
# 4) build supernetwork (which includes as a subprocess building unimodal graphs)
 # this saves as a .pkl file called G_super.pkl

# read pickled supernetwork (which includes unimodal networks joined by transfer edges)
# cwd = os.getcwd()
# with open(os.path.join(cwd, 'Data', 'Output_Data', 'G_super.pkl'), 'rb') as inp:
#     G_super = pickle.load(inp)
# print(G_super.fix_pre)
# print(G_super.coord_matrix.shape)

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
inv_nid_map = dict(zip(G_super_od.nid_map.values(), G_super_od.nid_map.keys()))   # also adjust the inverse nidmap

# build od_cnx and add to graph
od_cnx_edges = od_cnx(G_super_od, o_coord, d_coord)
od_cnx_edges = [(e[0], e[1], od_cnx_edges[e])for e in od_cnx_edges.keys()] # convert od_cnx_edges to proper form so that they can be added to graph
G_super_od.graph.add_edges_from(od_cnx_edges)

# then save the object
cwd = os.getcwd()
G_super_od.save_object(os.path.join(cwd, 'Data', 'Output_Data', 'G_super_od.pkl'))

#%%
# at this point, we have the supernetwork
# now convert to link cost file
df_link = nx.to_pandas_edgelist(G_super_od.graph)
# assign total cost as linear combination of all 5 cost factors
cost_factors = ['avg_TT_min', 'price', 'risk', 'reliability', 'discomfort']
cost_factor_cols = []
num_intervals = int(conf.config_data['Time_Intervals']['len_period'] / conf.config_data['Time_Intervals']['interval_spacing']) + 1
for c in cost_factors:
    for i in range(num_intervals):
        cost_factor_cols.append('interval'+str(i)+'_' + c)
cols_keep = ['source', 'target', 'mode_type'] + cost_factor_cols
df_link = df_link[cols_keep]
betas = conf.config_data['Beta_Params']
for i in range(num_intervals):
    df_link['interval'+str(i)+'_' + 'cost'] = (betas['b_TT'] * df_link['interval'+str(i)+'_' + 'avg_TT_min']
                                               + betas['b_disc'] * df_link['interval'+str(i)+'_' + 'discomfort']
                                               + betas['b_price'] * df_link['interval'+str(i)+'_' + 'price']
                                               + betas['b_rel'] * df_link['interval'+str(i)+'_' + 'reliability']
                                               + betas['b_risk'] * df_link['interval'+str(i)+'_' + 'risk'])
cost_cols = ['interval'+str(i)+'_' + 'cost' for i in range(num_intervals)]
df_linkcost = df_link[['source','target'] + cost_cols]
# add link id
df_linkcost['linkID'] = df_linkcost.index
# then make separate network topology file, called df_G
df_G = df_linkcost[['linkID', 'source', 'target']]


#%%
Gtest = nx.from_pandas_edgelist(df_linkcost, source='source', target='target', edge_attr=True)
#%%
djk_path = nx.dijkstra_path(Gtest, 'org', 'dst', 'interval5_cost')

#%% Prepare files for compatiblity with MAC-POSTS
# 1) Create graph topology file
filename = 'graph'
np.savetxt(filename, df_G, fmt='%d', delimiter=' ')
f = open(filename, 'r')
log = f.readlines()
f.close()
log.insert(0, 'EdgeId FromNodeId ToNodeId\n')
f = open(filename, 'w')
f.writelines(log)
f.close()

# 2) create link cost file 
filename = 'td_link_cost'
np.savetxt(filename, df_linkcost[['linkID'] + cost_cols], fmt='%d ' + (num_intervals-1)*'%f ' + '%f')
f = open(filename, 'r')
log = f.readlines()
f.close()
log.insert(0, 'link_ID td_cost\n')
f = open(filename, 'w')
f.writelines(log)
f.close()

# 3) # Create node cost df to prevent org-ps-ps or ps-ps-dst transfers
link_ID_map = dict(zip(tuple(zip(df_G['source'], df_G['target'])), df_G['linkID']))
node_costs = []
for n in list(G_super_od.nodes):
    edges_in = list(G_super_od.in_edges(n))
    edges_out = list(G_super_od.out_edges(n))
    for ei in edges_in:
        for eo in edges_out:                   
            # prevent consecutive transfers (so avoids ps-ps-ps or bs-ps-ps)
            if ((ut.mode(ei[0]), ut.mode(ei[1])) in G_super_od.pmx) & ((ut.mode(eo[0]), ut.mode(eo[1])) in G_super_od.pmx):
                node_costs.append([inv_nid_map[n], link_ID_map[ei], link_ID_map[eo]])
            # prevent od_cnx - transfer
            if (ei[0].startswith('org')) & ((ut.mode(eo[0]), ut.mode(eo[1])) in G_super_od.pmx):  
                node_costs.append([inv_nid_map[n], link_ID_map[ei], link_ID_map[eo]])
            # prevent transfer - od_cnx
            if ((ut.mode(ei[0]), ut.mode(ei[1])) in G_super_od.pmx) & (eo[1].startswith('dst')):  
                node_costs.append([inv_nid_map[n], link_ID_map[ei], link_ID_map[eo]])   

df_nodecost = pd.DataFrame(node_costs, columns = ['node_ID', 'in_link_ID', 'out_link_ID'])
for i in range(num_intervals):
    df_nodecost['interval'+str(i)+'_COST'] = 100000 # some arbitarily large number to prevent unwanted transfers

filename = 'td_node_cost'
np.savetxt(filename, df_nodecost, fmt='%d %d %d ' + (num_intervals-1)*'%f ' + '%f')
f = open(filename, 'r')
log = f.readlines()
f.close()
log.insert(0, 'node_ID in_link_ID out_link_ID td_cost\n')
f = open(filename, 'w')
f.writelines(log)
f.close()

# 4) edit the config file
# function to edit the config file for compatibility with MAC-POSTS
def edit_config(folder, graph_name, len_link_file, len_node_file):
    with open(folder + '/config.conf', 'w') as f:
        f.write('[Network] \n')
        f.write('network_name = ' + graph_name + '\n')
        f.write('num_of_link = ' + str(len_link_file) + '\n')
        f.write('num_of_node = ' + str(len_node_file) + '\n')
        #f.write('num_of_link = ' + str(df_linkcost.shape[0]) + '\n')
        #f.write('num_of_node = ' + str(df_nodecost.shape[0]))
        
folder = os.getcwd()
edit_config(folder, 'graph', df_linkcost.shape[0], df_nodecost.shape[0])

# 5) run shortest path
max_interval = num_intervals
num_rows_link_file = df_linkcost.shape[0]
num_rows_node_file = df_nodecost.shape[0]
link_cost_file_name = "td_link_cost"
node_cost_file_name = "td_node_cost"

dest_node_ID = inv_nid_map['dst']
origin_node_ID = inv_nid_map['org']



#%% **for visualization**: jitter the graphs and draw
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
