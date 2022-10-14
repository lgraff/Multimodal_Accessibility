#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 16:13:44 2022

@author: lindsaygraff
"""

# libraries and c
import numpy as np
import util_functions as ut
import config as conf

#%%

# inputs: graph, num of days of historical data, num of time intervals, num of scooter obs per time-interval day lower bound
# and upper bound, lower and upper bound of potential (x,y) coordinate of scooter, node id map, some cost parameters
# output: dict of dicts
def gen_data(G_superntwk, n_days, n_intervals, n_obs, bbox, od_cnx=False): #, avg_bike_segment_length):  
    # define the bounds
    xlb, xub, ylb, yub = bbox['minx'], bbox['maxx'], bbox['miny'], bbox['maxy']
    
    if not od_cnx:
        nid_map_fixed = G_superntwk.nid_map_fixed
    else: # only evaluate for org
        nid_map_fixed = {nid_num: nid_name for nid_num, nid_name in G_superntwk.nid_map.items() if nid_name in ['org']} 
    
    # initialize the scooter cost dictionary: key is the fixed node, the value is dict of costs (different cost for the different time intervals)
    all_costs = dict([(n, {}) for n in nid_map_fixed.values()])
    
    # For subsequent visualization purposes
    #fig, axs = plt.subplots(2, 5, sharex = True, sharey = True, figsize=(16,8))
    #plt.suptitle('Example: Scooter observations (red) for time interval 0 shown relative to fixed node bs1038 (blue)')
    # for subsequent plotting purposes 
    #node_coords = np.array([val for key,val in nx.get_node_attributes(G_u, 'pos').items() if key in list(node_id_map_fixed.values())])

    for i in range(n_intervals):  # each time interval 
        obs = {}  # obs is a dict, where the key is the day, the value is an array of coordinates representing different observations
        for j in range(n_days):  # each day
            n_obs = n_obs #np.random.uniform(n_obs_lb,n_obs_ub)  # how many scooter observations for the day-time interval pair
            # generate some random data: data is a coordinate matrix
            # the scooter observations should fit within the bounding box of the neighborhood mask polygon layer
            data = [(round(np.random.uniform(xlb, xub),8), 
                     round(np.random.uniform(ylb, yub),8)) for k in range(int(n_obs))]  
            obs[j] = np.array(data)  

        # find edge cost
        node_cost_dict = {}
        for n in nid_map_fixed.values():  # for each fixed node (or, for the org/dst when generating for od_cnx)
            all_min_dist = np.empty((1,n_days))  # initialize the min distance matrix, one entry per day
                       
            for d in range(n_days):  # how many days of historical scooter data we have 
                all_dist = ut.calc_great_circle_dist(np.array(G_superntwk.graph.nodes[n]['pos']), obs[d])  # dist from the fixed node to all observed scooter locations 
                min_dist = np.min(all_dist)  # choose the scooter with min dist. assume a person always walks to nearest scooter
                all_min_dist[0,d] = min_dist # for the given day, the dist from the fixed node to the nearest scooter is min_dist
                
#                 if (i == 0 and n == 'bs1038'):   # testing
#                     print(all_dist)
                # **********************************
                # JUST FOR VISUALIZATION PURPOSES
#                 # for fixed node bs1038 and time interval 0, visualize the scooter location data for each day
#                 if (i == 0 and n == 'bs1037'):
#                     row = d // 5
#                     col = d if d <=4 else (d-5)
#                     for k in range(len(obs[d][:,0])):
#                         axs[row,col].plot([G.nodes[n]['pos'][0], obs[d][k,0]], [G.nodes[n]['pos'][1], obs[d][k,1]], 
#                                  c='grey', ls='--', marker = 'o', mfc='r', zorder=1)
#                     axs[row,col].scatter(x = G.nodes[n]['pos'][0], y = G.nodes[n]['pos'][1], c='b', s = 200, zorder=2)
#                     axs[row,col].set_title('Day ' + str(d))
#                     axs[row,col].text(-79.93, 40.412, 'closest scooter: ' + str(round(min_dist,3)) + ' miles', ha='center')
# #                 # **********************************
            
            mean_min_dist = np.mean(all_min_dist)  # mean distance from node n to any scooter in past "n_days" days
            p95 = np.percentile(all_min_dist, 95)  # 95th percentile distance from node n to any scooter in past "n_days" days
            
            node_cost_dict[n] = {'interval'+str(i)+'_avg_TT_min': (mean_min_dist / conf.config_data['Speed_Params']['walk'] / 60),
                                 'interval'+str(i)+'_price': conf.config_data['Price_Params']['scoot']['fixed'],
                                 'interval'+str(i)+'_reliability': p95 / conf.config_data['Speed_Params']['walk'] / 60,
                                 'risk_idx': 1,
                                 'interval'+str(i)+'_risk': 1 * (mean_min_dist / conf.config_data['Speed_Params']['walk'] / 60),
                                 'mode_type':'w',
                                 'interval'+str(i)+'_discomfort': conf.config_data['Discomfort_Params']['scoot'] * (
                                     (mean_min_dist/conf.config_data['Speed_Params']['walk']/60)),
                                 'etype': 'transfer'}   

        for node, cost_dict in node_cost_dict.items():
            all_costs[node].update(cost_dict) 
    return all_costs

# num_days_of_data = 30
# num_obs = 100
# num_intervals = int(conf.config_data['Time_Intervals']['len_period'] / conf.config_data['Time_Intervals']['interval_spacing']) + 1
# # define which nodes are fixed and which come from flexible networks 
# all_fix_pre = ['bsd','ps','k', 'zd', 'kz']  # prefix for fixed nodes
# all_flex_pre = ['t', 'pb', 'pv', 'sc']  # prefix for flexible dropoff nodes
# G_super.separate_nidmap_fix_flex(all_fix_pre, all_flex_pre)

# sc_costs = gen_data(G_super, num_days_of_data, num_intervals, num_obs, bbox_study_area)


# #%%  observe
# import re
# test = list()
# for key in sc_costs.keys():
#     m = re.sub(r'[^a-zA-Z]', '', key)
#     test.append(m)
# print(set(test))


#%%



