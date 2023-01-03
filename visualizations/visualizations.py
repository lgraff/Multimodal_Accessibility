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
from od_connector import od_cnx
from shapely import wkt, geometry
from supernetwork import Supernetwork

#%% read pickled supernetwork (which includes unimodal networks joined by transfer edges)
cwd = os.getcwd()
G_super_od = od_cnx(os.path.join(cwd, 'Data', 'Output_Data', 'G_super.pkl'),
        conf.config_data['Supernetwork']['org'],[-79.938444, 40.418378])

#%%
# choose only a single neighborhood to use for visualization
cwd = os.getcwd()
path = os.path.join(cwd, 'Data', 'Input_Data', 'Neighborhoods', "Neighborhoods_.shp")
nhoods = gpd.read_file(path)  # all neighborhoods
hood = conf.config_data['Geography']['neighborhoods'][0]  # only the ith neighborhood in the list
hood_gdf = nhoods[nhoods['hood'] == hood]  

# one neighbrhood is still too big to visualize. let's scale down by x%
fig,ax = plt.subplots()
scale_factor = 0.7
x=0.1
y=-0.2
scaled_gdf = hood_gdf.geometry.scale(xfact=scale_factor, yfact=scale_factor).translate(-1/69*x, 1/69*y)
hood_gdf.plot(ax=ax, color='blue')
scaled_gdf.plot(ax=ax, color='green')
                           

df = pd.DataFrame.from_dict(dict(G_super_od.graph.nodes), orient="index").reset_index()
#gdf_pt = gpd.GeoDataFrame(data=df, geometry=df.pos)
df[['x','y']] = pd.DataFrame(df.pos.tolist())
gdf_nodes = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x,df.y))
gdf_nodes.set_crs(epsg='4326', inplace=True)
nodes_clip = gpd.clip(gdf_nodes, scaled_gdf)
nodes_clip.set_index('index', inplace=True)

# go back to the original PT graph, only keep nodes edges that are within the selected neighborhood
# 1) Nodes
G = nx.DiGraph()
node_dict = nodes_clip.to_dict(orient='index')
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

jitter_param_dict = {m: (j/200)*2 for j,m in enumerate(modes_included)}  # can adjust jitter param as necessary
# explicitly adjust org and dest if they are in the node list
if 'org' in list(G.nodes):
    G.nodes['org']['pos_adj'] = (G.nodes['org']['pos'][0], G.nodes['org']['pos'][1] + jitter_param_dict['pt']) 
if 'dst' in list(G.nodes):
    G.nodes['dst']['pos_adj'] = (G.nodes['dst']['pos'][0], G.nodes['dst']['pos'][1] + jitter_param_dict['sc']) 

# also include adjusted pos for org/dst
#G.nodes['org']['pos_adj'] = G.nodes['org']['pos']
#G.nodes['dst']['pos_adj'] = G.nodes['dst']['pos']

for m in modes_included:
    jitter_nodes(G, m, jitter_param_dict[m])

#G.remove_node('org')

# draw the graph in networkx
node_color_map = {'bs':'green', 'bsd':'lime', 'z':'blue', 'zd': 'skyblue', 'kz': 'dodgerblue',
                  'sc':'darkviolet', 't':'red', 'ps':'brown', 'rt': 'orange' , 'od':'black'}
node_color = [node_color_map[G.nodes[n]['node_type']] for n in G.nodes.keys()]
edge_color = ['darkgray' if G.edges[e]['mode_type'] == 'w' else 'black' for e in G.edges] 
edge_style = [(0,(5,10)) if G.edges[e]['mode_type'] == 'w' 
               else 'dotted' if G.edges[e]['mode_type'] in ['board','alight'] else 'solid' for e in G.edges] 

pos = 'pos_adj'
node_coords = nx.get_node_attributes(G, pos)    
fig, ax = plt.subplots(figsize=(20,20))
nx.draw_networkx_nodes(G, pos=node_coords, node_color=node_color, node_size=20, alpha=0.6, ax=ax)
nx.draw_networkx_edges(G, pos=node_coords, edge_color=edge_color, style=edge_style, alpha=0.6, arrowsize=10, ax=ax)
# add legend for node color    

# inv_node_cmap = dict(zip(node_color_map.values(), node_color_map.keys()))
# for v in set(inv_node_cmap.keys()):
#     ax.scatter([],[], c=v, label=inv_node_cmap[v])
# ax.legend(loc = 'upper right')

#nx.draw(G, pos=node_coords, with_labels=False, font_color='white',  font_weight = 'bold',
 #       node_size=15, node_color=node_color, edge_color=edge_color, alpha=0.7, arrowsize=10, style=edge_style, ax=ax)


#%%    
# plot for visualization
# node_color_map = {'bs':'green', 'bsd':'lime', 'z':'blue', 'zd': 'skyblue', 'kz': 'dodgerblue',
#                   'sc':'darkviolet', 't':'red', 'ps':'brown', 'rt': 'orange' , 'od':'black'}
# node_color = [node_color_map[G.nodes[n]['node_type']] for n in G.nodes.keys()]
# edge_color = ['darkgray' if G.edges[e]['mode_type'] == 'w' else 'black' for e in G.edges] 
# edge_style = [(0,(5,10)) if G.edges[e]['mode_type'] == 'w' 
#                else 'dotted' if G.edges[e]['mode_type'] in ['board','alight'] else 'solid' for e in G.edges] 
# ax = ut.draw_graph(G, node_color, node_color_map, edge_color, edge_style, adjusted=True)

# #%%
# for e in G.edges.keys():
#     if e[0].startswith('kz'):
#         print(e)


