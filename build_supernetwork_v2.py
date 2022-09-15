#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 15:35:18 2022

@author: lindsaygraff
"""

import util_functions as ut
import networkx as nx 
from build_unimodal_graphs import G_tnc, G_pv, G_pb, G_bs, G_pt, G_sc, G_z
import numpy as np
import config as conf

# this dict defines which graphs correspond to each mode type 
all_graphs_dict = {'t':G_tnc, 'pv':G_pv, 'pb':G_pb, 'bs':G_bs, 'pt':G_pt, 'sc':G_sc, 'z':G_z}

# this dict defines the node names corresponding to each mode type 
all_modes_nodes = {'bs':['bs', 'bsd'], 'pt':['ps','rt'], 't':['t'], 'sc':['sc'], 
                   'pv':['pv','k'], 'pb':['pb'], 'z':['zd','z','kz']}

# define which nodes are fixed and which come from flexible networks 
all_fix_pre = ['bsd','ps','k', 'zd', 'kz']  # prefix for fixed nodes
all_flex_pre = ['t', 'pb', 'pv', 'sc']  # prefix for flexible dropoff nodes

modes_included = conf.config_data['Supernetwork']['modes_included']

# this dict defines which modes and nodes are included in the supernetwork
modes_nodes_included = {k:v for k,v in all_modes_nodes.items() if k in modes_included}
networks_included = [all_graphs_dict[m] for m in modes_included]  # set([all_graphs_dict[m] for m in modes_included])

pmx = [('ps','ps'),('bsd','ps'),('ps','bsd'),('ps','t'),('t','ps'),('t','bsd'),('bsd','t'), # permitted mode change
       ('k','ps'),('k','t'),('k','bsd'),('ps','pb'),('pb','ps'),('ps','sc'),('sc','ps'),('k','sc'),
       ('bsd','sc'), ('sc','bsd'), ('ps','zd'), ('bsd','zd'), ('t','zd'), ('sc','zd'),
       ('kz','ps'),('kz','t'),('kz','bsd'),('kz','sc')]  
#G_u = nx.union_all(networks_included)

#%%
class Supernetwork:
    def __init__(self, graphs_included, fix_pre, flex_pre):
        self.networks = graphs_included
        self.graph = nx.union_all(graphs_included)
        self.fix_pre = fix_pre  # which modes are fixed in the supernetwork
        self.flex_pre = flex_pre   # which modes are flex in the supernewtork
    
    # print the graphs that are incuded in the network
    def print_mode_types(self):
        print(self.networks)
    
    # add the matrix that contains the coordinates of each node in the supernetwork
    def add_coord_matrix(self):
        coords_dict = nx.get_node_attributes(self.graph, 'pos')
        nid_map = dict(zip(range(len(coords_dict.keys())), list(coords_dict.keys())))
        coord_matrix = np.array(list(coords_dict.values()))
        #self.coord_dict= coords_dict
        self.nid_map = nid_map
        self.coord_matrix = coord_matrix
        #return (nid_map, coord_matrix)
    
    # add the matrix that contains the great circle distance between all pairs of nodes
    def add_gcd_dist_matrix(self):
        # gcd distance matrix: gcd_dist[i,j] is the great circle distance from node i to node j
        gcd_dist = np.empty([len(self.nid_map), len(self.nid_map)])  # great circle dist in meters 
        for i in range(gcd_dist.shape[0]):
            dist_to_all_nodes = ut.calc_great_circle_dist(self.coord_matrix[i], self.coord_matrix)  # calc distance from node i to all other nodes
            gcd_dist[i] = dist_to_all_nodes  # calc_great_circle_dist is a function defined above 
        self.gcd_dist = gcd_dist
        #return gcd_dist
        
    # separate the node ID map by fixed nodes and flexible nodes
    def separate_nidmap_fix_flex(self):
        self.nid_map_fixed = {key:val for key,val in self.nid_map.items() if ut.mode(val) in self.fix_pre}
        self.nid_map_flex = {key:val for key,val in self.nid_map.items() if ut.mode(val) in self.flex_pre}
    
    # define the permitted mode changes within the supernetwork
    # pmx is defined in the config file
    def define_pmx(self, pmx):
        self.pmx = pmx
    
    def add_transfer_edges(self, W, scoot_cost_data):
        etype = 'transfer'
        trans_edges = {}
        
        for i in list(self.nid_map_fixed.keys()):
            #attrs = {}
            i_name = self.nid_map[i]  # map back to node name
            catch = ut.wcz(i, self.gcd_dist, W)  # catchment zone around i (includes both fixed and flex nodes)
            # build fixed-fixed transfer edge
            for j in catch:
                if j in self.nid_map_fixed.keys():
                    j_name = self.nid_map[j]  # map back to node name
                    if (ut.mode(i_name), ut.mode(j_name)) in self.pmx:         # if mode switch allowed between i and j
                        # build the transfer edge
                        edge = (i_name, j_name)
                        # find the walking time associated with transfer edge, call it walk_cost
                        walk_time = self.gcd_dist[i,j] / conf.config_data['Speed_Params']['walk'] / 60  # dist[m] / speed [m/s] / 60 s/min  --> [min]
                        wait_time = 0
                        # account for a no-cost public transit transfer (actually this creates errors, need to use node-movement cost)
                        #fixed_price = -PT_price if (mode(i_name) == 'ps') & (mode(j_name) == 'ps') else 0
                        fixed_price = 0
                        
                        # also add an inconvenience cost; this needs to be more carefully considered. make sufficiently large for now
                        # actually we can embed inconvenience cost into "discomfort" attribute
                        attr_dict = {'avg_TT_min': walk_time + wait_time,
                                     'price': fixed_price,
                                     'reliability': walk_time * conf.config_data['Reliability_Params']['walk'],
                                     'risk_idx': 1,  # for now, just assume all transfers associated with risk = 1
                                     'risk': 1 * (walk_time + wait_time),
                                     'discomfort': conf.config_data['Discomfort_Params']['walk'] * walk_time,
                                     'type': etype,
                                     'mode_type': 'w'}                
                        trans_edges[edge] = attr_dict                      
                
            # find the nearest neighbor in each flex network
            # but first, remove 'pv' nodes to prevent arbitary transfers to the pv network
            # 'pv' nodes are considered flexible for the sake of OD connectors  
            flex_pre_tx = self.flex_pre.copy()
            if 'pv' in self.flex_pre:
                flex_pre_tx.remove('pv') 
            
            if flex_pre_tx:  # check that list is not NoneType
                for m in flex_pre_tx:    # transfers from fixed-flex or flex-fixed
                    if ((ut.mode(i_name), m) in self.pmx) | ((m, ut.mode(i_name)) in self.pmx):    # if mode switch allowed between i and m
                        nnID, nnName, nnDist = ut.nn(i, self.gcd_dist, m, self.nid_map)  # tuple in the form of (node_id, node_name, dist)
                        if nnID in catch:   #and (m != 'sc')):
                            k_name = nnName
                            edge_in = (i_name, k_name)                
                            edge_out = (k_name, i_name)
                            walk_time = nnDist / conf.config_data['Speed_Params']['walk'] / 60  # dist[m] / speed [m/s] / 60 s/min               
                            wait_time = conf.config_data['Speed_Params']['TNC']['wait_time'] if ut.mode(edge_in[1]) == 't' else 0
                            fixed_price = conf.config_data['Price_Params']['TNC']['fixed'] if ut.mode(edge_in[1]) == 't' else 0
        
                            if m != 'sc':   # includes inconvenience cost
                                # the transfer edge cost is constant for all times 
                                
                                # separately check if edge_in / edge_out is allowed
                                if (ut.mode(i_name), ut.mode(k_name)) in self.pmx:   
                                    attr_dict = {'avg_TT_min': walk_time + wait_time,
                                                 'price': fixed_price,
                                                 'reliability': walk_time * conf.config_data['Reliability_Params']['walk'],
                                                 'risk_idx': 1,  # for now, just assume all transfers associated with risk = 1
                                                 'risk': 1 * ( walk_time + wait_time),
                                                 'discomfort': conf.config_data['Discomfort_Params']['walk'] * walk_time,
                                                 'type': etype,
                                                 'mode_type': 'w'} 
                                    trans_edges[edge_in] = attr_dict
                                    
                                if (ut.mode(k_name), ut.mode(i_name)) in self.pmx:  # prevents (tnc-parking) transfers
                                    #if mode(k_name) == 't':  # this is a cheap solution, should think of better way if possible
                                    # remove wait time 
                                    # Hardwiring avg_TT_min: walk_time and price:0 is a cheap solution
                                    # The reason I'm doing this is to avoid paying fixed or waiting costs when tranferring TO tnc
                                    # need to think of a more elegant way. tbd 
                                    attr_dict = {'avg_TT_min': walk_time,
                                                 'price': 0,
                                                 'reliability': walk_time *  conf.config_data['Reliability_Params']['walk'],
                                                 'risk_idx': 1,  # for now, just assume all transfers associated with risk = 1
                                                 'risk': 1 * walk_time,
                                                 'discomfort': conf.config_data['Discomfort_Params']['walk'] * walk_time,
                                                 'type': etype,
                                                 'mode_type': 'w'} 
                                    trans_edges[edge_out] = attr_dict  
        
                            else:  # mode is scooter
                                # when transferring TO scooter, assign costs that were created above according to historical data
                                if (ut.mode(i_name), ut.mode(k_name)) in self.pmx:
                                    scoot_attr_dict = scoot_cost_data[i_name]
                                    #attr_dict =  {key: (val + inc_cost) for key, val in attr_dict.items()}  # add inconvenience cost to the cost of walking to scooter
                                    scoot_attr_dict['type'] = etype
                                    trans_edges[edge_in] = scoot_attr_dict
        
                                # when transferring FROM scooter, use nearest neighbor distance b/c ride scooter to closest pickup point for next mode
                                if (ut.mode(k_name), ut.mode(i_name)) in self.pmx:  
                                    attr_dict = {'avg_TT_min': walk_time,
                                                 'price': fixed_price,
                                                 'reliability': walk_time * conf.config_data['Reliability_Params']['walk'],
                                                 'risk_idx': 1,  # for now, just assume all transfers associated with risk = 1
                                                 'risk': 1 * walk_time,
                                                 'discomfort': conf.config_data['Discomfort_Params']['walk'] * walk_time,
                                                 'type': etype,
                                                 'mode_type': 'w'} 
                                    trans_edges[edge_out] = attr_dict        
    
        self.transfer_edges = trans_edges  # for testing purposes so we can access the tx edges directly
        # now add the transfer edges to the supernetwork
        self.graph.add_edges_from(trans_edges)
#%%        
G_super = Supernetwork(networks_included, all_fix_pre, all_flex_pre)
G_super.print_mode_types()
G_super.add_coord_matrix()
G_super.add_gcd_dist_matrix()
G_super.separate_nidmap_fix_flex()
G_super.define_pmx(pmx)

#%%
# test: add transfer edges
W = 0.5 * conf.config_data['Conversion_Factors']['meters_in_mile']
# first get get scooter data
from data_gen_functions import gen_data
num_days_of_data = 30
num_obs = 100
num_intervals = int(conf.config_data['Time_Intervals']['len_period'] / conf.config_data['Time_Intervals']['interval_spacing']) + 1
sc_costs = gen_data(G_super, num_days_of_data, num_intervals, num_obs, conf.bbox_study_area)

#%%
# then add transfer edges 
G_super.add_transfer_edges(W, sc_costs)
#G_super.gcd_dist[:3,:3]

#%%
import re
mode_pre = []
for n in G_super.graph.nodes:
    m = re.sub(r'[^a-zA-Z]', '', n)
    mode_pre.append(m)

print(set(mode_pre))

#%%
mode_pre = []
for n in sc_costs.keys():
    m = re.sub(r'[^a-zA-Z]', '', n)
    mode_pre.append(m)
print(set(mode_pre))



