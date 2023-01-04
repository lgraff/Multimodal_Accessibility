
# description of file:

# libraries 
import networkx as nx
import pickle
import os
import re
import config as conf
import util_functions as ut
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from od_connector import od_cnx

# compile supernetwork with od-connectors
# function od_cnx takes the supernetwork as input, then output the supernetwork with od connectors
def compile_G_od(path):
    G_od = od_cnx(path,conf.config_data['Supernetwork']['org'],conf.config_data['Supernetwork']['dst'])
    return G_od

# create time-dependent cost dfs for the individual cost components
# output is a dict of the form: {cost_component: td_cost_df}
# the cost df has dimensions (num_links x (3 + num_time_intervals)), wheret the 3 add'l columns are due to presence of source, target, mode_type
# the (i+3)th column is the cost attribute associated with the ith departure interval
def create_td_cost_arrays(G_super_od):
    # convert graph to df of link costs
    df_link = nx.to_pandas_edgelist(G_super_od.graph)
    cost_factors = ['avg_TT_sec', 'price', 'risk', 'reliability', 'discomfort']
    # check that lal columns are filled out -- complete
    # for c in cost_factors:
    #     col_name = '0_'+c
    #     #print(df_link[col_name].isna().sum())

    # Here we build the travel time multiplier as a function of time 
    # some arbitary linear function
    len_period = int(conf.config_data['Time_Intervals']['len_period'])
    num_intervals = int(conf.config_data['Time_Intervals']['len_period'] / conf.config_data['Time_Intervals']['interval_spacing']) + 1
    print(num_intervals)
    x = np.linspace(0, len_period, num_intervals )  # x is time [min past] relative to 07:00 AM
    m = (1.5-1)/len_period # slope
    y = m*x + 1
    # plt.plot(x, y, 'o', color='black', zorder=2);
    # plt.plot(x, y, color='red', zorder=1);
    # plt.xlabel('Time (seconds relative to 07:00AM)')
    # plt.ylabel('Travel time multiplier \n (relative to baseline)')

    cost_factors = ['avg_TT_sec', 'price', 'risk', 'reliability', 'discomfort']
    cost_factors_0 = ['0_' + c for c in cost_factors]  # _0 represents the 0th departure time interval
    cols_keep = ['source', 'target', 'mode_type'] + cost_factors_0
    df_link = df_link[cols_keep]
    cost_factor_cols = [str(i) +'_' + c for c in cost_factors for i in range(1,num_intervals)]

    cost_attr_df_dict = {'avg_TT_sec':(), 'price':(), 'reliability':(), 'discomfort':(), 'risk':()}

    # get all travel time columns representing the TT of the link for different departure time intervals
    # result is a df with the following cols: source, target, mode_type, 0_avg_TT_sec, 1_avg_TT_sec, 2_avg_TT_sec...
    # where i_avg_TT_sec represents the avg amount of time it takes to cross the link at the start of the ith departure interval
    cost_attr_name = 'avg_TT_sec'
    df_var = df_link[['source','target','mode_type','0_'+cost_attr_name]].copy()
    interval_cols = [str(n) + '_' + cost_attr_name for n in range(0,num_intervals)]
    static_tt_modes = ['bs','w','pb','sc','t_wait','alight']   # static modes are those whose travel tiem is independent of traffice (i.e. bike/scooter/walk)
    dynamic_tt_modes = ['pv','z','t','pt','board']   # dynamic modes are those whose travel time depends on traffic (i.e. vehicle modes)
    tt_multiplier = y.copy()
    df_var_all = pd.DataFrame()

    for m in dynamic_tt_modes:
        df_m = df_var[df_var['mode_type'] == m].copy()
        data = [tt_multiplier[idx] * df_m['0_' + cost_attr_name] for idx in range(len(interval_cols))]  # multiply the base TT by the TT muiltiplier
        data = pd.concat(data, axis=1).set_axis(labels=interval_cols, axis=1)
        data = pd.concat([df_m[['source','target','mode_type']], data], axis=1)
        df_var_all = pd.concat([df_var_all, data], axis=0)
    for m in static_tt_modes:
        df_m = df_var[df_var['mode_type'] == m].copy()
        data = [df_m['0_' + cost_attr_name]] *  len(interval_cols)  # repeat the same value 
        data = pd.concat(data, axis=1).set_axis(labels=interval_cols, axis=1)
        data = pd.concat([df_m[['source','target','mode_type']], data], axis=1)
        df_var_all = pd.concat([df_var_all, data], axis=0)
    cost_attr_df_dict['avg_TT_sec'] = df_var_all.copy() 

    # get all price columns
    # result is a df with the following cols: source, target, mode_type, 0_price, 1_price, 2_price...
    # where i_price represents the price it takes to cross the link at the start of the ith departure interval
    cost_attr_name = 'price'
    df_var = df_link[['source','target','mode_type', '0_avg_TT_sec', '0_'+cost_attr_name]].copy()
    interval_cols = [str(n) + '_' + cost_attr_name for n in range(0,num_intervals)]
    fixed_price_modes = ['pv','pb', 'pt', 'board', 'alight', 'w','t_wait']    # modes that have a fixed price
    usage_price_modes = ['z','t','bs','sc']    # modes whose price depends on usage time 
    usage_prices = conf.config_data['Price_Params']
    usage_prices = {'z':usage_prices['zip']['ppmin'], 't':usage_prices['TNC']['ppmin'], 'bs':usage_prices['bs']['ppmin'],
                    'sc':usage_prices['scoot']['ppmin']}
    df_var_all = pd.DataFrame()

    for m in usage_price_modes:
        df_m = df_var[df_var['mode_type'] == m].copy()
        # TODO: for each unimodal graph, check that '0_price' is just the fixed price 
        data = [df_m['0_' + 'price'] +   # fixed mileage component (for TNC) + price for usage time
                    tt_multiplier[idx] * usage_prices[m] / 60 * df_m['0_avg_TT_sec'] for idx in range(len(interval_cols))]
        data = pd.concat(data, axis=1).set_axis(labels=interval_cols, axis=1)
        data = pd.concat([df_m[['source','target','mode_type']], data], axis=1)
        df_var_all = pd.concat([df_var_all, data], axis=0)
    for m in fixed_price_modes:
        df_m = df_var[df_var['mode_type'] == m].copy()
        data = [df_m['0_' + cost_attr_name]] *  len(interval_cols)  # repeat the same value 
        data = pd.concat(data, axis=1).set_axis(labels=interval_cols, axis=1)
        data = pd.concat([df_m[['source','target','mode_type']], data], axis=1)
        df_var_all = pd.concat([df_var_all, data], axis=0)

    cost_attr_df_dict['price'] = df_var_all.copy()

    # get all reliability columns
    modes = df_link['mode_type'].unique().tolist()
    cost_attr_name = 'reliability'
    df_var = df_link[['source','target','mode_type', '0_avg_TT_sec', '0_'+cost_attr_name]].copy()
    interval_cols = [str(n) + '_' + cost_attr_name for n in range(0,num_intervals)]
    rel_weights = conf.config_data['Reliability_Params']
    rel_weights = {'pt':rel_weights['PT_traversal'], 'board':rel_weights['PT_wait'], 't':rel_weights['TNC'],
                    'pb':rel_weights['pb'], 'pv':rel_weights['pv'], 'sc':rel_weights['scoot'], 'bs':rel_weights['bs'],
                    'z':rel_weights['zip'], 'w':rel_weights['walk'], 't_wait':rel_weights['TNC_wait'],
                    'sc_wait': 1.25,  # put this in the config file
                    'alight':1}
    df_var_all = pd.DataFrame()  

    # special case: scooter transfers 
    # we do the soooter transfers separately becasue we already generated 95th percentile TT, hence we will not create it as rel_weight * avgTT as we do below for the other modes
    df_sc_tx = df_var[(df_var['mode_type'] == 'w') & (df_var['target'].str.startswith('sc'))].copy() # filter by rows going TO scooter
    data = [df_sc_tx['0_' + cost_attr_name]] *  len(interval_cols)
    data = pd.concat(data, axis=1).set_axis(labels=interval_cols, axis=1)
    data = pd.concat([df_sc_tx[['source','target','mode_type']], data], axis=1)
    df_var_all = pd.concat([df_var_all, data], axis=0)  

    # note: we are using df_TT because it already has the time cost for each interval
    df_all_other = cost_attr_df_dict['avg_TT_sec'][~((cost_attr_df_dict['avg_TT_sec']['mode_type'] == 'w') & (cost_attr_df_dict['avg_TT_sec']['target'].str.startswith('sc')))]
    for m in modes:
        # reliability is defined as reliability_coef * avg_tt, for all intervals
        df_m = df_all_other[df_all_other['mode_type'] == m].copy()
        data = [rel_weights[m] * df_m[str(i) + '_avg_TT_sec'] for i in range(len(interval_cols))]
        data = pd.concat(data, axis=1).set_axis(labels=interval_cols, axis=1)
        data = pd.concat([df_m[['source','target','mode_type']], data], axis=1)
        df_var_all = pd.concat([df_var_all, data], axis=0)

    cost_attr_df_dict['reliability'] = df_var_all.copy()
    # TODO: right now, tnc waiting time is constant by departure time. could think about changing it 

    # add discomfort and risk. assume constant by departure interval
    for c in ['discomfort', 'risk']:
        df_var = df_link[['source','target','mode_type','0_' + c]].copy()
        interval_cols = [str(n) + '_' + c for n in range(0,num_intervals)]
        data = [df_var['0_' + c]] *  len(interval_cols)
        data = pd.concat(data, axis=1).set_axis(labels=interval_cols, axis=1)
        data = pd.concat([df_var[['source','target','mode_type']], data], axis=1)
        cost_attr_df_dict[c] = data

    for df in cost_attr_df_dict.values():
        df.sort_values(by=['source','target','mode_type'], inplace=True)
    
    return cost_attr_df_dict

# returns the node cost df where each row has the following form: node_id, in_linkID, out_linkID, cost
def create_node_cost_file(G_super_od, link_id_map, inv_nid_map):
    link_id_map = link_id_map #dict(zip(tuple(zip(df_G['source'], df_G['target'])), df_G['linkID']))
    node_costs = []
    for n in list(G_super_od.graph.nodes):
        edges_in = list(G_super_od.graph.in_edges(n))
        edges_out = list(G_super_od.graph.out_edges(n))
        for ei in edges_in:
            ei_num = (inv_nid_map[ei[0]], inv_nid_map[ei[1]])
            for eo in edges_out:            
                eo_num = (inv_nid_map[eo[0]], inv_nid_map[eo[1]])
                # prevent consecutive transfers (so avoids ps-ps-ps or bs-ps-ps)
                if ((ut.mode(ei[0]), ut.mode(ei[1])) in G_super_od.pmx) & ((ut.mode(eo[0]), ut.mode(eo[1])) in G_super_od.pmx):
                    node_costs.append([inv_nid_map[n], link_id_map[ei_num], link_id_map[eo_num], 'double_tx'])
                # prevent od_cnx - transfer
                if (ei[0].startswith('org')) & ((ut.mode(eo[0]), ut.mode(eo[1])) in G_super_od.pmx):  
                    node_costs.append([inv_nid_map[n], link_id_map[ei_num], link_id_map[eo_num], 'double_tx'])
                # prevent transfer - od_cnx
                if ((ut.mode(ei[0]), ut.mode(ei[1])) in G_super_od.pmx) & (eo[1].startswith('dst')):  
                    node_costs.append([inv_nid_map[n], link_id_map[ei_num], link_id_map[eo_num], 'double_tx'])   
                # alternatively, we can write this in one if statement *i think*: prevent two consecutive walking edges
                # if G_super_od.graph.edges[ei]['mode_type'] == 'w' & G_super_od.graph.edges[eo]['mode_type'] == 'w' 
                
                # prevent transfer - tw - t
                if (G_super_od.graph.edges[ei]['mode_type'] == 'w') & (eo[1].startswith('tw')):
                    node_costs.append([inv_nid_map[n], link_id_map[ei_num], link_id_map[eo_num], 'double_tx'])
                # add a -2.75 cost to account for fee-less PT transfers
                if (n.startswith('ps')) & (ei[0].startswith('ps')) & (G_super_od.graph.edges[eo]['mode_type'] == 'board'):
                    node_costs.append([inv_nid_map[n], link_id_map[ei_num], link_id_map[eo_num], 'pt_tx'])

    df_nodecost = pd.DataFrame(node_costs, columns = ['node_ID', 'in_link_ID', 'out_link_ID','type'])
    return df_nodecost

# cwd = os.getcwd()
# G_super_od = compile_G_od(os.path.join(cwd, 'Data', 'Output_Data', 'G_super.pkl'))
# # adjust the inverse nidmap
# inv_nid_map = dict(zip(G_super_od.nid_map.values(), G_super_od.nid_map.keys()))   

# betas = conf.config_data['Beta_Params']  
# num_intervals = int(conf.config_data['Time_Intervals']['len_period'] / conf.config_data['Time_Intervals']['interval_spacing']) + 1
# cost_attr_df_dict = create_td_cost_arrays(G_super_od)  # call the function, which returns a dict of form {attr_name: attr_cost_df}
# # also create the travel time df in units of "n" second multiples
# n = conf.config_data['Time_Intervals']['interval_spacing']
# interval_cols = [str(n) + '_' + 'avg_TT_sec' for n in range(0,num_intervals)]
# df_tt_multiple = pd.DataFrame(np.around(cost_attr_df_dict['avg_TT_sec'][interval_cols].to_numpy() / n), columns=interval_cols).astype('int')
# df_tt_multiple = pd.concat([cost_attr_df_dict['avg_TT_sec'][['source','target','mode_type']], df_tt_multiple] ,axis=1)

# # represent the cost arrays in numpy so that each one can be multiplied by the corresponding beta
# cost_arrays = {c:() for c in cost_attr_df_dict.keys()}
# for c, df in cost_attr_df_dict.items():
#     cost_arrays[c] = df.drop(['source','target','mode_type'], axis=1).to_numpy()

# # ********************************* START NEW PROGRAM HERE **************************
# # **THIS IS THE STEP THAT MATTERS FOR SENSITIVITY ANALYSIS     
# cost_final = (betas['b_TT'] * cost_arrays['avg_TT_sec'] + betas['b_disc'] * cost_arrays['discomfort'] + betas['b_price'] * cost_arrays['price'] 
#                 + betas['b_rel'] * cost_arrays['reliability'] + betas['b_risk'] * cost_arrays['risk'])


# # back to pandas: map the source, target pair to its numerical representation. then append to the left of the array
# # map alphanumeric node names to their numeric names
# df_G = cost_attr_df_dict['avg_TT_sec'][['source','target']].applymap(lambda x: inv_nid_map[x])
# df_G.insert(0, 'linkID', df_G.index)

# #%% Prepare files for compatiblity with MAC-POSTS
# # 1) Create graph topology file
# folder = os.path.join(os.getcwd(), 'macposts_files')
# filename = 'graph'
#np.savetxt(os.path.join(folder, filename), df_G, fmt='%d', delimiter=' ')

# # 2) create link cost file td_link_cost
# filename = 'td_link_cost'
# linkID_array = df_G['linkID'].to_numpy().reshape((cost_final.shape[0],1))
# td_link_cost = np.hstack((linkID_array, cost_final))

# # 3) time-dep travel time for each link and node (just TT, not full cost), in units of 10 second multiples
# # i.e. if the TT is 12 seconds, the cell entry is 12/10 = 1.2 ~= 1 (we round to the nearest integer) 
# # td_link_tt
# filename = 'td_link_tt'
# td_link_tt = np.hstack((linkID_array, df_tt_multiple.drop(columns=['source','target','mode_type']).to_numpy()))
# td_link_tt[:, 1:][td_link_tt[:, 1:] <= 0] = 1

# #4) create node cost file td_node_cost
# df_nodecost = create_node_cost_file(G_super_od, df_G)
# nodecosts = np.full((len(df_nodecost), num_intervals), 500)
# nodecost_ids = df_nodecost.to_numpy()
# td_node_cost = np.empty((df_nodecost.shape[0], 3 + num_intervals))

# td_node_cost[:,:3] = nodecost_ids
# td_node_cost[:,3:] = nodecosts
# filename = 'td_node_cost'
# # save td_link_cost, td_link_tt, and td_node_cost as a compressed numpy zip archive
# folder = os.path.join(os.getcwd(),'macposts_files', 'macposts_arrays.npz')
# np.savez_compressed(folder, td_link_cost=td_link_cost, td_link_tt=td_link_tt, td_node_cost=td_node_cost)

