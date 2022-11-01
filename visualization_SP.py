#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 15:34:26 2022

@author: lindsaygraff
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 11:20:58 2022

@author: lindsaygraff
"""

# libraries
import os
import re
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import util_functions as ut
import config as conf
import pickle
import pandas as pd
import geopandas as gpd
import math
from shapely import wkt, geometry
from supernetwork import Supernetwork

#%% read pickled supernetwork (which includes unimodal networks joined by transfer edges)
cwd = os.getcwd()
with open(os.path.join(cwd, 'Data', 'Output_Data', 'G_super_od.pkl'), 'rb') as inp:
    G_super_od = pickle.load(inp)
#%%
# choose all neighborhoods for visualization
cwd = os.getcwd()
path = os.path.join(cwd, 'Data', 'Input_Data', 'Neighborhoods', "Neighborhoods_.shp")
nhoods = gpd.read_file(path)  # all neighborhoods

hood = conf.config_data['Geography']['neighborhoods']# only the ith neighborhood in the list
hood_gdf = nhoods[nhoods['hood'].isin(hood)]  
                           

df = pd.DataFrame.from_dict(dict(G_super_od.graph.nodes), orient="index").reset_index()
#gdf_pt = gpd.GeoDataFrame(data=df, geometry=df.pos)
df[['x','y']] = pd.DataFrame(df.pos.tolist())
gdf_nodes = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x,df.y))
gdf_nodes.set_crs(epsg='4326', inplace=True)

# go back to the original PT graph, only keep nodes edges that are within the selected neighborhood
# 1) Nodes
G = nx.DiGraph()
node_dict = gdf_nodes.to_dict(orient='index')
G.add_nodes_from(node_dict.keys())
nx.set_node_attributes(G, node_dict)
# 2) Edges
df_edges = nx.to_pandas_edgelist(G_super_od.graph)
df_edges_keep = df_edges.loc[(df_edges['source'].isin(list(G.nodes))) & (df_edges['target'].isin(list(G.nodes)))]
df_edges_keep.set_index(['source','target'], inplace=True)
edge_dict = df_edges_keep.to_dict(orient='index')
G.add_edges_from(edge_dict.keys())
nx.set_edge_attributes(G, edge_dict)

#%% **for visualization**: jitter the graphs and draw
def jitter_nodes(G, network_type, jitter_param):
    #G_adj = G.copy()
    #print(G.nodes) #[node]['pos'])
    # adjust the nodes positions in the copy 
    nodes = [n for n in G.nodes.keys() if n not in ['org','dst'] if G.nodes[n]['nwk_type'] == network_type] 
    for n in nodes:
        adj_x = G.nodes[n]['pos'][0] + jitter_param
        adj_y = G.nodes[n]['pos'][1] + jitter_param
        nx.set_node_attributes(G, {n: {'pos_adj':(adj_x, adj_y)}})

#%%
# adjust the graphs in the supernetwork using the jitter function
# this is strictly for plotting
# for j, m in enumerate(modes_included):
#     print(j,m)
#     #if m != 't':  # tnc graph is the "base" road network so it will not be adjusted
#     G_adj = all_graphs_dict[m].copy()
#     jitter_nodes(G_adj, jitter_param=(j/100)*2)
#     adjusted_graph_dict[m] = G_adj

modes_included = conf.config_data['Supernetwork']['modes_included']
modes_included.remove('pt')
modes_included.insert(math.floor(len(modes_included)/2), 'pt')  # explicitly have PT graph be the one in the center

inv_nid_map = dict(zip(G_super_od.nid_map.values(), G_super_od.nid_map.keys()))   # also adjust the inverse nidmap

# previously j/200
jitter_param_dict = {m: (j/50)*2 for j,m in enumerate(modes_included)}  # can adjust jitter param as necessary
# explicitly adjust org and dest if they are in the node list
G.nodes[inv_nid_map['org']]['pos_adj'] = (G.nodes[inv_nid_map['org']]['pos'][0] + 0.1*jitter_param_dict['pt'], 
                                          G.nodes[inv_nid_map['org']]['pos'][1] + jitter_param_dict['pt']) 
G.nodes[inv_nid_map['dst']]['pos_adj'] = (G.nodes[inv_nid_map['dst']]['pos'][0] + 0.3*jitter_param_dict['sc'], 
                                          G.nodes[inv_nid_map['dst']]['pos'][1] + jitter_param_dict['sc']) 

#%% sp edges
#sp_nodestring = '7781 4337 4493 4641 3822 4297 4522 3877 4621 4519 4007 4197 3802 4397 4858 4423 4193 4967 4351 4219 4775 4882 3921 4983 4080 4294 4895 4394 4535 4960 4592 3905 4270 4126 3882 4655 4134 4926 4803 3925 4444 4202 3972 4271 3924 4135 4619 3914 7290 7299 7288 7316 7330 7597 7016 6971 7782'
#sp_nodes = sp_nodestring.replace(' ', ',')

sp_nodes =  # INSERT PATH HERE
sp_edges = list(zip(sp_nodes[:-1], sp_nodes[1:]))
    
# also include adjusted pos for org/dst
#G.nodes['org']['pos_adj'] = G.nodes['org']['pos']
#G.nodes['dst']['pos_adj'] = G.nodes['dst']['pos']

for m in modes_included:
    jitter_nodes(G, m, jitter_param_dict[m])

# draw the graph in networkx
node_color_map = {'bs':'green', 'bsd':'lime', 'z':'blue', 'zd': 'skyblue', 'kz': 'dodgerblue',
                  'sc':'darkviolet', 't':'red', 'ps':'brown', 'rt': 'orange' , 'od':'black'}
node_color = [node_color_map[G.nodes[n]['node_type']] for n in G.nodes.keys()]
edge_color = ['darkgray' if G.edges[e]['mode_type'] == 'w' else 'black' for e in G.edges] 
edge_style = [(0,(5,10)) if G.edges[e]['mode_type'] == 'w' 
               else 'dotted' if G.edges[e]['mode_type'] in ['board','alight'] else 'solid' for e in G.edges] 

pos = 'pos_adj'
node_coords = nx.get_node_attributes(G, pos)    
fig, ax = plt.subplots(figsize=(40,40))
nx.draw_networkx(G, pos=node_coords, node_color=node_color, node_size=30, alpha=0.6, 
                 edgelist=sp_edges, edge_color='black', width=5, ax=ax)
# add legend for node color    
inv_node_cmap = dict(zip(node_color_map.values(), node_color_map.keys()))
for v in set(inv_node_cmap.keys()):
    ax.scatter([],[], c=v, label=inv_node_cmap[v])
ax.legend(loc = 'lower right', fontsize=40, markerscale=3)

plt.save_fig(os.path.join(cwd, 'graph_sp_ex1.png'), format='PNG')
#%%
#nx.draw_networkx_edges(G, pos=node_coords, edge_color=edge_color, style=edge_style, alpha=0.6, arrowsize=10, ax=ax)



#nx.draw(G, pos=node_coords, with_labels=False, font_color='white',  font_weight = 'bold',
 #       node_size=15, node_color=node_color, edge_color=edge_color, alpha=0.7, arrowsize=10, style=edge_style, ax=ax)

