
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
nid_map = G_super_od.nid_map
inv_nid_map = dict(zip(nid_map.values(), nid_map.keys()))   
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

# ************************************************************************
#%% # 2) create node cost file td_node_cost
df_nodecost = macposts.create_node_cost_file(G_super_od, link_id_map).sort_values(by='type') # nodeID, inLinkID, outLinkID, cost
nodecosts_dbltx = np.full((len(df_nodecost[df_nodecost['type'] == 'double_tx']), num_intervals), 500)   # replicate the node cost for each departure interval
nodecosts_pttx = np.full((len(df_nodecost[df_nodecost['type'] == 'pt_tx']), num_intervals), round(-2.75/2, 2)) 
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

#%% **THIS BEGINS THE SENSITIVITY ANALYSIS for scooter pricing**     
betas['b_rel'] = 5/3600
betas['b_TT'] = 10/3600
betas['b_risk'] = 0.1
betas['b_disc'] = 0.1

# read in arrays
files = np.load(os.path.join(cwd, 'macposts_files.npz'))
td_link_tt = files['td_link_tt']
td_node_cost = files['td_node_cost']
interval_sec = conf.config_data['Time_Intervals']['interval_spacing']
timestamp = int((60/interval_sec) * 30) # minute 30 of the hour-long interval 

sc_links = [lid for link, lid in link_id_map.items() if nid_map[link[0]].startswith('sc') and nid_map[link[1]].startswith('sc')]

all_paths = {} # form is {(beta_param_1, beta_param_2...): link_sequence}
for sc_ppmin in [0.09, 0.14, 0.19, 0.24, 0.29, 0.34, 0.39]:
    # reduce the scooter cost by x percent
    price_reduction_pct = (0.39 - sc_ppmin) / 0.39  # percent reduction in scoot link cost
    cost_array_price = cost_arrays['price'].copy()
    cost_array_price[sc_links] = (1-price_reduction_pct) * cost_array_price[sc_links] 
    # final link cost array is a linear combination of the individual components
    cost_final = (betas['b_price'] * cost_array_price + betas['b_TT'] * cost_arrays['avg_TT_sec'] + betas['b_disc'] * cost_arrays['discomfort'] + 
                    + betas['b_rel'] * cost_arrays['reliability'] + betas['b_risk'] * cost_arrays['risk'])

    # Prepare additional files for compatiblity with MAC-POSTS
    # 4) create link cost file td_link_cost
    filename = 'td_link_cost'
    td_link_cost = np.hstack((linkID_array, cost_final))
    cost_array_dict = {'td_link_cost': td_link_cost, 'td_link_tt': td_link_tt, 'td_node_cost':td_node_cost}
    # run tdsp from macposts
    macposts_folder = os.path.join(cwd, 'macposts_files')
    #timestamp = 5
    tdsp_array = tdsp.tdsp_macposts(macposts_folder, cost_array_dict, inv_nid_map, 5) # shortest path (node sequence)
    # get path info
    nid_map = G_super_od.nid_map
    node_seq = tdsp_array[:,0]
    link_seq = tdsp_array[:-1,1]
    path = [nid_map[int(n)] for n in node_seq]
    print(path)
    all_paths[sc_ppmin] = (node_seq, link_seq, path)
    # release the memory associated with large link cost array
    del tdsp_array
    del cost_final
    del td_link_cost
    del cost_array_dict
    #del cost_array_price
    gc.collect()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            #%% Get the totals of the individual cost components
path_costs = {} # this will be a dict of dicts
for sc_ppmin, (node_seq, link_seq, path) in all_paths.items():
    price_total, risk_total, rel_total, tt_total, discomfort_total = 0, 0, 0, 0, 0
    cost_total = 0
    price_reduction_pct = (0.39 - sc_ppmin) / 0.39  # percent reduction in scoot link cost
    cost_array_price = cost_arrays['price'].copy()
    cost_array_price[sc_links] = (1-price_reduction_pct) * cost_array_price[sc_links] 
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
                nodecost = -2.75 #round(-2.75/2, 2)  # ideally would be -2.75, but in that case we are finding that it is advantageous (under b_TT = b_rel = 0) to transfer 4 times (even though -2.75 should apply only once)
            else:
                nodecost = 0
        else:
            nodecost = 0
        # td_node_cost: nodeID, inLinkID, outLinkID
        # get the price, risk, and reliability of the link at timestamp t
        price_link, risk_link, rel_link, tt_link = cost_array_price[l,t], cost_arrays['risk'][l,t], cost_arrays['reliability'][l,t], cost_arrays['avg_TT_sec'][l,t]  # (these arrays do not have a col for linkID)
        discomfort_link = cost_arrays['discomfort'][l,t]
        #cost_link = td_link_cost[l,t+1]  # cannot use td_link_cost b/c it only reflects most recently used betas     
        # update time and cost totals
        price_total += (price_link + nodecost)
        risk_total += risk_link
        rel_total += rel_link
        tt_total += tt_link
        discomfort_total += discomfort_link
        #cost_total += cost_link
        cost_attributes = {'price_total': price_total, 'risk_total': risk_total, 'rel_total':round(rel_total/60,2), 
                            'tt_total':round(tt_total/60,2), 'discomfort_total': discomfort_total}
        t = t + intervals_cross  
        #print(nid_map[node_in], nid_map[node_out], price_link)
        #del cost_array_price
        gc.collect()
    path_costs[sc_ppmin] = cost_attributes   
# %% Just print the path 
for params, link_seq in all_paths.items():
    print('path parms:', params)
    for l in link_seq:
        l = int(l)
        node_in, node_out = inv_link_id_map[l]
        print(nid_map[node_in], nid_map[node_out])
    print('********************')

# %%
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sc_prices = list(path_costs.keys())
cost_dict = list(path_costs.values())

all_prices = [d['price_total'] for d in cost_dict]
all_risk = [d['risk_total'] for d in cost_dict]
all_rel = [d['rel_total'] for d in cost_dict]
all_tt = [d['tt_total'] for d in cost_dict]
all_disc = [d['discomfort_total'] for d in cost_dict]

x_axis = np.array(sc_prices)
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax2.plot(x_axis, all_prices, label='price', color='C0',  zorder=0)
#ax2.plot(x_axis[7:13], all_prices[7:13], color='C0',  zorder=0)
ax2.plot(x_axis, all_risk, label='risk', color='C1',  zorder=0)
ax1.plot(x_axis, all_rel, label='reliability', color='C4',  zorder=1)
ax1.plot(x_axis, all_tt, label='travel time', color='C2',  zorder=0)
ax2.plot(x_axis, all_disc, label='discomfort', color='C3',  zorder=0)

ax2.scatter(x_axis, all_prices, s=20, color='C0',  zorder=0)
#ax2.plot(x_axis[7:13], all_prices[7:13], color='C0',  zorder=0)
ax2.scatter(x_axis, all_risk, s=20, color='C1',  zorder=0)
ax1.scatter(x_axis, all_rel,  s=20, color='C4',  zorder=0)
ax1.scatter(x_axis, all_tt,  s=20, color='C2',  zorder=0)
ax2.scatter(x_axis, all_disc,  s=20, color='C3',  zorder=0)

# note where the mode shift occurs
ax1.axvspan(0.14, 0.19, alpha=0.5, color='grey')
#plt.axvline(x=6, color='black',linestyle='--',linewidth=2)
#plt.axvline(x=12, color='black',linestyle='--',linewidth=2)
# distinguish region by mode type
ax2.text(0.085, 35,'Scooter &',fontsize='medium', zorder=1)
ax2.text(0.09, 32.5,'Walking',fontsize='medium', zorder=1)
ax2.text(0.29, 35,'TNC &',fontsize='medium', zorder=1)
ax2.text(0.285, 32.5,'Walking',fontsize='medium', zorder=1)
# ax2.text(15.5, 33,'TNC &',fontsize='large', zorder=1)
# ax2.text(15.2, 30,'Walking',fontsize='large', zorder=1)
ax1.set_ylabel('Travel Time (min), Reliability (min)')
ax2.set_ylabel('Price ($), Risk, Discomfort')
ax1.set_xlabel(r'Scooter price per minute (\$/min)')   #$\beta_{TT}$')
ax2.legend(loc='upper right')
ax1.legend(loc='upper left').set_zorder(100)
ax2.set_xticks(np.arange(0.09,0.40,0.05), fontsize=10)
ax1.set_yticks(np.arange(0,60,5), fontsize=6)
ax2.set_yticks(np.arange(0,80,5), fontsize=6)

ax1.set_zorder(ax1.get_zorder()+1)
ax1.patch.set_visible(False)
plt.show(0)
# %%
