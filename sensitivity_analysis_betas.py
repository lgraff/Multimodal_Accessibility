
#%%# libraries 
import networkx as nx
import pickle
import os
import re
import gc
import config as conf
import util_functions as ut
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import create_macposts_files as macposts
import tdsp_macposts as tdsp
cwd = os.getcwd()

# compile the od graph
G_super_od = macposts.compile_G_od(os.path.join(cwd, 'Data', 'Output_Data', 'G_super.pkl'))

# get the inverse nidmap
inv_nid_map = dict(zip(G_super_od.nid_map.values(), G_super_od.nid_map.keys()))   
# get parameters
betas = conf.config_data['Beta_Params']  
num_intervals = int(conf.config_data['Time_Intervals']['len_period'] / conf.config_data['Time_Intervals']['interval_spacing']) + 1
# call the user-defined function "create_td_cost_arrays", which returns a dict of form {attr_name: attr_cost_df}
# where attr_cost_df has dimensions num_links x num_intervals 
cost_attr_df_dict = macposts.create_td_cost_arrays(G_super_od)  
# also create the travel time df in units of "n" second multiples
n = conf.config_data['Time_Intervals']['interval_spacing']
interval_cols = [str(n) + '_' + 'avg_TT_sec' for n in range(0,num_intervals)]
df_tt = cost_attr_df_dict['avg_TT_sec'].reset_index(drop=True)
df_tt_multiple = pd.DataFrame(np.around(df_tt[interval_cols].to_numpy() / n), columns=interval_cols).astype('int')
df_tt_multiple = pd.concat([df_tt[['source','target','mode_type']], df_tt_multiple] ,axis=1)

# represent the cost arrays in numpy so that each one can be multiplied by the corresponding beta
cost_arrays = {}  #{c:() for c in cost_attr_df_dict.keys()}
for c, df in cost_attr_df_dict.items():
    cost_arrays[c] = df.drop(['source','target','mode_type'], axis=1).to_numpy()

# Prepare files for compatiblity with MAC-POSTS
# 1) Create graph topology file
# back to pandas: map the source, target pair to its numerical representation. then append to the left of the array
# map alphanumeric node names to their numeric names
df_G = df_tt[['source','target']].applymap(lambda x: inv_nid_map[x])
df_G.reset_index(inplace=True, drop=True)
df_G.insert(0, 'linkID', df_G.index)  # add link ID
folder = os.path.join(cwd, 'macposts_files')
filename = 'graph'
np.savetxt(os.path.join(folder, filename), df_G, fmt='%d', delimiter=' ')
# add a header EdgeId	FromNodeId	ToNodeId
f = open(os.path.join(folder, filename), 'r')
log = f.readlines()
f.close()
log.insert(0, 'EdgeId	FromNodeId	ToNodeId\n')
f = open(os.path.join(folder, filename), 'w')
f.writelines(log)
f.close()

# match a link ID to a (source, target) pair
link_id_map = dict(zip(tuple(zip(df_G['source'], df_G['target'])), df_G['linkID']))
inv_link_id_map = dict(zip(link_id_map.values(), link_id_map.keys()))
linkID_array = df_G['linkID'].to_numpy().reshape((df_G.shape[0],1))

#%% # 2) create node cost file td_node_cost
df_nodecost = macposts.create_node_cost_file(G_super_od, link_id_map, inv_nid_map).sort_values(by='type') # nodeID, inLinkID, outLinkID, cost
nodecosts_dbltx = np.full((len(df_nodecost[df_nodecost['type'] == 'double_tx']), num_intervals), 500)   # replicate the node cost for each departure interval
nodecosts_pttx = np.full((len(df_nodecost[df_nodecost['type'] == 'pt_tx']), num_intervals), -2.75) 
nodecost_ids = df_nodecost[['node_ID', 'in_link_ID', 'out_link_ID']].to_numpy()
td_node_cost = np.empty((df_nodecost.shape[0], 3 + num_intervals))

td_node_cost[:,:3] = nodecost_ids
td_node_cost[:,3:] = np.vstack((nodecosts_dbltx, nodecosts_pttx))
filename = 'td_node_cost'
# save td_link_cost, td_link_tt

# 3) time-dep travel time for each link and node (just TT, not full cost), in units of 10 (or some other #) second multiples
# i.e. if the TT is 12 seconds, the cell entry is 12/10 = 1.2 ~= 1 (we round to the nearest integer) 
# td_link_tt
filename = 'td_link_tt'
td_link_tt = np.hstack((linkID_array, df_tt_multiple.drop(columns=['source','target','mode_type']).to_numpy()))
td_link_tt[:, 1:][td_link_tt[:, 1:] <= 0] = 1

# save cost_arrays, td_node_cost, td_link_tt for later use
np.savez_compressed(os.path.join(cwd, 'macposts_files'), td_link_tt=td_link_tt, td_node_cost=td_node_cost)
del td_node_cost

#%% **THIS BEGINS THE SENSITIVITY ANALYSIS**     
#betas['b_rel'] = 10/3600
#betas['b_TT'] = 0/3600
betas['b_risk'] = 0.1
#betas['b_disc'] = 0.1
betas['b_rel'] = 10/3600

# read in arrays
files = np.load(os.path.join(cwd, 'macposts_files.npz'))
td_link_tt = files['td_link_tt']
td_node_cost = files['td_node_cost']
interval_sec = conf.config_data['Time_Intervals']['interval_spacing']
timestamp = int((60/interval_sec) * 30) # minute 30 of the hour-long interval 

all_paths = {} # form is {(beta_param_1, beta_param_2...): link_sequence}
for b_TT in np.arange(0/3600, 21/3600, 2.5/3600): #[20/3600]: #
    betas['b_TT'] = b_TT
    for b_risk in np.arange(0, 0.2, 0.05): # [100/3600]: #
        betas['b_risk'] = b_risk
        # final link cost array is a linear combination of the individual components
        cost_final = (betas['b_TT'] * cost_arrays['avg_TT_sec'] + betas['b_disc'] * cost_arrays['discomfort'] + betas['b_price'] * cost_arrays['price'] 
                        + betas['b_rel'] * cost_arrays['reliability'] + betas['b_risk'] * cost_arrays['risk'])

        # Prepare additional files for compatiblity with MAC-POSTS
        # 4) create link cost file td_link_cost
        filename = 'td_link_cost'
        td_link_cost = np.hstack((linkID_array, cost_final))
        cost_array_dict = {'td_link_cost': td_link_cost, 'td_link_tt': td_link_tt, 'td_node_cost':td_node_cost}
        # run tdsp from macposts
        macposts_folder = os.path.join(cwd, 'macposts_files')
        tdsp_array = tdsp.tdsp_macposts(macposts_folder, cost_array_dict, inv_nid_map, timestamp) # shortest path (node sequence)
        # get path info
        nid_map = G_super_od.nid_map
        node_seq = tdsp_array[:,0]  # 
        link_seq = tdsp_array[:-1,1]
        path = [nid_map[int(n)] for n in node_seq]
        print(path)
        all_paths[(round(b_TT,5), round(b_risk,5))] = (node_seq, link_seq)
        # release the memory associated with large link cost array
        del tdsp_array
        del cost_final
        del td_link_cost
        del cost_array_dict
        gc.collect()

#%% Get the totals of the individual cost components
path_costs = {} # this will be a dict of dicts
for beta_params, (node_seq, link_seq) in all_paths.items():
    price_total, risk_total, rel_total, tt_total = 0, 0, 0, 0
    cost_total = 0
    t = timestamp
    for idx, l in enumerate(link_seq):  # the link seq
        # look up how many time intervals it takes to cross the link
        l = int(l) 
        # adjustment of time (check on this i.e. can delete?)
        if t >= num_intervals:
            t = num_intervals - 1 
        intervals_cross = td_link_tt[l,t+1]  # add one bc first col is linkID
        node_in, node_out = inv_link_id_map[l]   
        # TODO: figure out how to add node costs (below needs to be checked)
        if idx > 0:
            node_id = node_seq[idx]
            in_link_id = int(link_seq[idx-1])
            out_link_id = l
            if any(np.equal(nodecost_ids,[node_id, in_link_id, out_link_id]).all(1)):
                nodecost = -2.75
            else:
                nodecost = 0
        else:
            nodecost = 0
        # td_node_cost: nodeID, inLinkID, outLinkID
        # get the price, risk, and reliability of the link at timestamp t
        price_link, risk_link, rel_link, tt_link = cost_arrays['price'][l,t], cost_arrays['risk'][l,t], cost_arrays['reliability'][l,t], cost_arrays['avg_TT_sec'][l,t]  # (these arrays do not have a col for linkID)
        #cost_link = td_link_cost[l,t+1]  # cannot use td_link_cost b/c it only reflects most recently used betas     
        # update time and cost totals
        price_total += (price_link) #+ nodecost)
        risk_total += risk_link
        rel_total += rel_link
        tt_total += tt_link
        #cost_total += cost_link
        cost_attributes = {'price_total': price_total, 'risk_total': risk_total, 'rel_total':rel_total, 'tt_total':tt_total}
        t = t + intervals_cross  
        #print(nid_map[node_in], nid_map[node_out], risk_link)
    path_costs[beta_params] = cost_attributes

#TODO: assign bus alighting edge a risk of 1 (account of risk of being hit by car)  
#TODO: also an idea: every 5 (10?) min of walking is associated with 1 unit of risk idx
# probably you will cross a road every 5 min? idk       

# %% Just print the path 
for params, (node_seq, link_seq) in all_paths.items():
    print('path parms:', params)
    for l in link_seq:
        l = int(l)
        node_in, node_out = inv_link_id_map[l]
        print(nid_map[node_in], nid_map[node_out])
    print('********************')

# %%
