#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 11:04:55 2022

@author: lindsaygraff
"""

# libraries
import os
import re
import numpy as np
import pickle
import copy
import util_functions as ut
import config as conf
from data_gen_functions import gen_data

# test od coord

# this function builds OD connectors on the fly
# input: graphs, coordinates of the org and dst, the nodeID map, list of fixed nodes, list of flex modes, 
# gc dist between all nodes in the graph, max walk distance W, and walk speed parameter 
# output: graphs with OD connector edges added, along with their associated cost 

# note that there are some exceptions: 
# 1) org connects to flex PV;  2) org does not connect to fixed parking;  3) dst does not connect to flex PV
# 4) dst connects to fixed parking;  5) dst does not connect to fixed zip depot
def od_cnx(G_super_filepath, o_coord, d_coord):
    #cwd = os.getcwd()
    #with open(os.path.join(cwd, 'Data', 'Output_Data', 'G_super.pkl'), 'rb') as inp:
     #   G_super = pickle.load(inp)

    with open(G_super_filepath, 'rb') as inp:
        G_super = pickle.load(inp)

    # define od pair
    o_coord = o_coord  #[-79.94868171046522, 40.416379503934145]
    d_coord = d_coord  #[-79.91944888168011, 40.45228774674678]
    # build od coord matrix
    od_coords = np.array([o_coord, d_coord])
    # add the org and dst nodes to the graph along with their positions 
    G_super.graph.add_nodes_from([('org', {'pos': tuple(o_coord), 'nwk_type':'od', 'node_type':'od'}),  
                                     ('dst', {'pos': tuple(d_coord), 'nwk_type':'od', 'node_type':'od'})])

    # 2 rows because 1 origin and 1 destination 
    nrow, ncol = 2, len(G_super.nid_map) + 2  # add 2 to len(nid_map) to account for addition of org node and dst node

    gcd_od = np.empty([nrow, ncol])  
    for j in range(nrow):  # first the org, then the dest
        dist_to_map_nodes = ut.calc_great_circle_dist(od_coords[j], G_super.coord_matrix) # dist from OD to modal graph nodes
        dist_to_od = ut.calc_great_circle_dist(od_coords[j], od_coords) # dist from O to O and O to D, then D to O and D to D
        gcd_od[j] = np.hstack((dist_to_map_nodes, dist_to_od)) # horizontally concatenate 
        
    # now add gcd_dist_od to the original gcd_dist matrix
    # result is gcd_dist_all, which contains the gcd dist b/w all pairs of nodes in the graph, including org and dst
    gcd_dist_all = np.vstack((G_super.gcd_dist, gcd_od[:,:len(G_super.nid_map)]))
    gcd_dist_all = np.hstack((gcd_dist_all, np.transpose(gcd_od)))

    # adjust the nid_map and gcd_dist matrix to reflect the addition of org and dst
    G_super.nid_map[max(G_super.nid_map.keys())+1] = 'org'
    G_super.nid_map[max(G_super.nid_map.keys())+1] = 'dst'
    G_super.gcd_dist = gcd_dist_all
    inv_nid_map = dict(zip(G_super.nid_map.values(), G_super.nid_map.keys()))   # also adjust the inverse nidmap

    # build od_cnx and add to graph
    #od_cnx_edges = od_cnx(G_super, o_coord, d_coord) 
    #od_cnx_edges = [(e[0], e[1], od_cnx_edges[e])for e in od_cnx_edges.keys()] # convert od_cnx_edges to proper form so that they can be added to graph
    #G_super.graph.add_edges_from(od_cnx_edges)
    # then save the object
    #cwd = os.getcwd()
    #G_super.save_object(os.path.join(cwd, 'Data', 'Output_Data', 'G_super_od.pkl'))

    # first get get scooter data
    num_days_of_data = conf.config_data['Scoot_Data_Generation']['num_days_of_data']
    num_obs = conf.config_data['Scoot_Data_Generation']['num_obs']
    num_intervals = int(conf.config_data['Time_Intervals']['len_period'] / conf.config_data['Time_Intervals']['interval_spacing']) + 1
    sc_costs = gen_data(G_super, num_days_of_data, num_intervals, num_obs, od_cnx=True) # conf.bbox_study_area, )

    # then build o/d connectors
    od_cnx_edges = {}
    inv_nid_map = dict(zip(G_super.nid_map.values(), G_super.nid_map.keys())) 
        
    for i_name in ['org','dst']:
        print(inv_nid_map[i_name])
        #***** I think we should not be using G_super.gcd_dist, but rather a newly created one which includes the ODs and their gcd dists
        W_od = conf.config_data['Supernetwork']['W_od'] * conf.config_data['Conversion_Factors']['meters_in_mile']
        catch = ut.wcz(inv_nid_map[i_name], G_super.gcd_dist, W_od)  # find WCZ
        #print('-----')
        # build od connector edge for each FIXED node in the catchment zone 
        for j in catch:
            if j in G_super.nid_map_fixed.keys():
                j_name = G_super.nid_map[j]  # map back to node name
                #print(j_name, mode(j_name))
                if ((i_name == 'org' and ut.mode(j_name) == 'k') | (i_name == 'org' and ut.mode(j_name) == 'kz') | (i_name == 'dst' and ut.mode(j_name) == 'zd')):   
                    continue  # exceptions 2 and 5
                if i_name == 'org':
                    edge = (i_name, j_name)  # build org connector
                if i_name == 'dst':
                    edge = (j_name, i_name)  # build dst connector     
                
                walk_time = (G_super.gcd_dist[inv_nid_map[i_name], j] / 
                             conf.config_data['Speed_Params']['walk']) / 60  # walking traversal time [min] of edge
                wait_time = 0 
                # note that wait time is not included. this is because we're dealing with fixed modes. only TNC has wait time
                # wait time for PT is embedded in alighting edges
                fixed_price = 0  # for all fixed modes
                attr_dict = {'avg_TT_min': walk_time + wait_time,
                             'price': fixed_price,
                             'reliability': (walk_time + wait_time) * conf.config_data['Reliability_Params']['walk'],
                             #'risk_idx': 1,  # for now, just assume all transfers associated with risk = 1,
                             'risk': 1 * (walk_time + wait_time),
                             'discomfort': conf.config_data['Discomfort_Params']['walk'] * walk_time}
                             #'type': 'od_cnx',
                             #'mode_type': 'w'}                
                od_cnx_edges[edge] = (ut.time_dep_attr_dict(attr_dict, int(conf.config_data['Time_Intervals']['len_period'] / conf.config_data['Time_Intervals']['interval_spacing']) + 1)
                                        | {'mode_type':'w'}
                                        | {'type':'od_cnx'})
                         
                
        # some fixed modes do not have a node within the wcz. for these modes, we will instead connect the 
        # org or dst to the nearest neighbor node of for these specific fixed modes. consider this like 
        # relaxing the wcz constraint
        catch_node_names = [G_super.nid_map[c] for c in catch]
        catch_fixed_modes = [re.sub(r'[^a-zA-Z]', '', cname) for cname in catch_node_names]
        #print(catch_fixed_modes)
        
        # which fixed mode does not have a node in the wcz?
        rem_fixed_modes = set(G_super.flex_pre) - set(catch_fixed_modes)
        if rem_fixed_modes:
            for rm in rem_fixed_modes:
                if ((i_name == 'org' and rm == 'k') | (i_name == 'org' and rm == 'kz') | (i_name == 'dst' and rm == 'zd')):   # exceptions 2/5
                    continue
                # nn calc
                nnID, nnName, nnDist = ut.nn(inv_nid_map[i_name], G_super.gcd_dist, rm, G_super.nid_map) 
                r_name = nnName
                cnx_edge_length = nnDist
                walk_time = cnx_edge_length / conf.config_data['Speed_Params']['walk'] / 60  # [min]
                wait_time = 0
                fixed_price = 0

                attr_dict = {'avg_TT_min': walk_time + wait_time,
                             'price': fixed_price,
                             'reliability': (walk_time + wait_time) * conf.config_data['Reliability_Params']['walk'],
                             #'risk_idx': 1,  # for now, just assume all transfers associated with risk = 1,
                             'risk': 1 * (walk_time + wait_time),
                             'discomfort': conf.config_data['Discomfort_Params']['walk'] * walk_time}
                             #'type': 'od_cnx',
                             #'mode_type': 'w'}                     

                if i_name == 'org':
                    edge = (i_name, r_name)  # build org connector
                if i_name == 'dst':
                    edge = (r_name, i_name)  # build dst connector

                od_cnx_edges[edge] = (ut.time_dep_attr_dict(attr_dict, int(conf.config_data['Time_Intervals']['len_period'] / conf.config_data['Time_Intervals']['interval_spacing']) + 1)
                                        | {'mode_type':'w'}
                                        | {'type':'od_cnx'})      

        # build od connector edge for the nearest flexible node (relax constraints that needs to be in wcz)
        # also includes an org connector from org to nearest PV node, but does NOT include a dst connector from dst to nearest PV node
        # PV = personal vehicle 
        for m in G_super.flex_pre:
            if i_name == 'dst' and m == 'pv':  # exception 3
                continue
            nnID, nnName, nnDist = ut.nn(inv_nid_map[i_name], G_super.gcd_dist, m, G_super.nid_map) 
            k_name = nnName
            #print(k_name)
            #if nnID in catch:   *we have decided to relax this constraint*
            
            # do this separately for scooters/TNCs and other flex modes. but have not yet generated TNC data to account for variable pickup wait times
            if (i_name == 'org' and m == 'sc'):
                edge = (i_name, k_name)
                #print(edge)
                sc_cost_dict = sc_costs[i_name]
                sc_cost_dict['type'] = 'od_cnx'    # store edge type
                od_cnx_edges[edge] = sc_cost_dict

            else:   
                cnx_edge_length = nnDist
                #print(i_name)
                #print(nnName, nnDist)
                walk_time = cnx_edge_length / conf.config_data['Speed_Params']['walk'] / 60  # [min]
                #print(walk_time)
                wait_time = conf.config_data['Speed_Params']['TNC']['wait_time'] if (m == 't') & (i_name == 'org') else 0
                fixed_price = conf.config_data['Price_Params']['TNC']['fixed'] if (m == 't') & (i_name == 'org') else 0
                rel_TNC_wait = conf.config_data['Reliability_Params']['TNC_wait'] 
                # to do: add zip fixed price, maybe 9/4                          
                
                attr_dict = {'avg_TT_min': walk_time + wait_time,
                             'price': fixed_price,
                             'reliability': (walk_time * conf.config_data['Reliability_Params']['walk'] 
                                             + wait_time * rel_TNC_wait),
                             #'risk_idx': 1,  # for now, just assume all transfers associated with risk = 1,
                             'risk': 1 * (walk_time + wait_time),
                             'discomfort': conf.config_data['Discomfort_Params']['walk'] * walk_time}
                             #'type': 'od_cnx',
                             #'mode_type': 'w'}  
                
                if i_name == 'org':
                    edge = (i_name, k_name)  # build org connector
                if i_name == 'dst':
                    edge = (k_name, i_name)  # build dst connector
                
                od_cnx_edges[edge] = (ut.time_dep_attr_dict(attr_dict, int(conf.config_data['Time_Intervals']['len_period'] / conf.config_data['Time_Intervals']['interval_spacing']) + 1)
                                        | {'mode_type':'w'}
                                        | {'type':'od_cnx'})       
    od_cnx_edges = [(e[0], e[1], od_cnx_edges[e])for e in od_cnx_edges.keys()] # convert od_cnx_edges to proper form so that they can be added to graph
    G_super.graph.add_edges_from(od_cnx_edges)
    return G_super
    # then save the object
    #G_super.save_object(G_od_filepath) #os.path.join(cwd, 'Data', 'Output_Data', #'G_super_od.pkl'))


