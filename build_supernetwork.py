#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 11:52:05 2022

@author: lindsaygraff
"""

# libraries
import os
from build_unimodal_graphs import G_tnc, G_pv, G_pb, G_bs, G_pt, G_sc, G_z
print('unimodal graphs are built')
import config as conf
#import util_functions as ut
import supernetwork as sn

#%%
def build_supernetwork(output_fpath):
# this dict defines which graphs correspond to each mode type 
    all_graphs_dict = {'t':G_tnc, 'pv':G_pv, 'pb':G_pb, 'bs':G_bs, 'pt':G_pt, 'sc':G_sc, 'z':G_z}
    
    # this dict defines the node names corresponding to each mode type 
    all_modes_nodes = {'bs':['bs', 'bsd'], 'pt':['ps','rt'], 't':['t'], 'sc':['sc'], 
                       'pv':['pv','k'], 'pb':['pb'], 'z':['zd','z','kz']}
    
    # define which nodes are fixed and which come from flexible networks 
    all_fix_pre = ['bsd','ps','k', 'zd', 'kz']  # prefix for fixed nodes
    all_flex_pre = ['t', 'pb', 'pv', 'sc']  # prefix for flexible dropoff nodes
    
    modes_included = conf.config_data['Supernetwork']['modes_included']
    fix_pre_included = [n for m in modes_included for n in all_modes_nodes[m] if n in all_fix_pre]
    flex_pre_included = [n for m in modes_included for n in all_modes_nodes[m] if n in all_flex_pre]
    
    # this dict defines which modes and nodes are included in the supernetwork
    modes_nodes_included = {k:v for k,v in all_modes_nodes.items() if k in modes_included}
    networks_included = [all_graphs_dict[m] for m in modes_included]  # set([all_graphs_dict[m] for m in modes_included])
    
    pmx = [('ps','ps'),('bsd','ps'),('ps','bsd'),('ps','t'),('t','ps'),('t','bsd'),('bsd','t'), # permitted mode change
           ('k','ps'),('k','t'),('k','bsd'),('ps','pb'),('pb','ps'),('ps','sc'),('sc','ps'),('k','sc'),
           ('bsd','sc'), ('sc','bsd'), ('ps','zd'), ('bsd','zd'), ('t','zd'), ('sc','zd'),
           ('kz','ps'),('kz','t'),('kz','bsd'),('kz','sc')]  
    
    # initiailize the network
    G_super = sn.Supernetwork(networks_included, fix_pre_included, flex_pre_included)
    G_super.print_mode_types()
    G_super.add_coord_matrix()
    G_super.add_gcd_dist_matrix()
    G_super.separate_nidmap_fix_flex()
    G_super.define_pmx(pmx)
    
    # add transfer edges
    W_tx = conf.config_data['Supernetwork']['W_tx'] * conf.config_data['Conversion_Factors']['meters_in_mile']
    G_super.add_transfer_edges(W_tx)
    #G_super.gcd_dist[:3,:3]
            
    G_super.save_object(output_fpath)
    return G_super
    #cwd = os.getcwd()
    #ut.save_object(G_super, os.path.join(cwd, 'Data', 'Output_Data', 'G_super.pkl'))

# test the function
cwd = os.getcwd()


#%% build supernetwork, also save as pickled object for later use if necessary (avoid compiling it many times)
G_super = build_supernetwork(os.path.join(cwd, 'Data', 'Output_Data', 'G_super.pkl')) 
