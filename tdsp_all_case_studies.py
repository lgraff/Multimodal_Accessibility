#%% libraries
import MNMAPI
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

cwd = os.getcwd()
# create config file
# function to edit the config file for compatibility with MAC-POSTS
def edit_config(folder, graph_name, num_links, num_nodes):
    with open(folder + '/config.conf', 'w') as f:
        f.write('[Network] \n')
        f.write('network_name = ' + graph_name + '\n')
        f.write('num_of_link = ' + str(num_links) + '\n')
        f.write('num_of_node = ' + str(num_nodes) + '\n')

def tdsp_macposts_all_times(macposts_folder, cost_arrays, inv_nid_map, start_time_min):
    time_start = 7*3600 # seconds start after midnight (7am)
    time_end = 9*3600 # seconds end after midnight (9am)
    interval_spacing = 10 # seconds
    num_intervals = int((time_end-time_start) / interval_spacing)

    timestamp = int(60/interval_spacing * start_time_min)  # start at 7:30am
    #end_timestamp = # end at 8:45am
    intervals_keep = [i for i in range(timestamp, num_intervals)]  # keep only results from 8-9am
    td_link_cost = cost_arrays['td_link_cost'][:, [0]+intervals_keep].copy().astype(float)  # add one for link id
    td_link_tt = cost_arrays['td_link_tt'][:, [0]+intervals_keep].copy().astype(float)
    td_node_cost = cost_arrays['td_node_cost'][:, [0,1,2]+intervals_keep].copy().astype(float)  # add three for in_node, pass-through_node, end_node

    num_rows_link_file = td_link_cost.shape[0]
    num_rows_node_file = cost_arrays['td_node_cost'].shape[0]
    nodes = list(inv_nid_map.values())
    num_nodes = len(nodes)
    max_interval = td_link_cost.shape[1] - 1  # subtract one for the link_id

    # edit the config file
    edit_config(macposts_folder, 'graph', num_rows_link_file, num_nodes)

    # invoke TDSP api from mac-posts
    tdsp_api = MNMAPI.tdsp_api()
    tdsp_api.initialize(macposts_folder, max_interval, num_rows_link_file, num_rows_node_file)
    tdsp_api.read_td_cost_py(td_link_tt, td_link_cost, td_node_cost, td_node_cost)  # assume td_node_cost = td_node_tt
    print('TDSP api has successfully read the files')

    # find tdsp
    tdsp_api.build_tdsp_tree(inv_nid_map['dst'])
    # get the sp every 60 seconds for 75 minutes (from 7:30am-8:45am)
    intervals_in_min = int(60/interval_spacing)  # number of time intervals in 1 minute
    new_timestamp = 0  # b/c we have zero-indexed the 8am timestamp
    tdsp_dict = {}
    for i in range(76):
        tdsp_arr = tdsp_api.extract_tdsp(inv_nid_map['org'], new_timestamp)
        tdsp_dict[timestamp + new_timestamp] = tdsp_arr 
        new_timestamp += intervals_in_min
    return(tdsp_dict) 


#%%
# read in the supernetwork
with open('supernetwork.pkl', 'rb') as inp:
    G_super_od = pickle.load(inp)
# get the node_id map 
nid_map = G_super_od.nid_map
inv_nid_map = dict(zip(nid_map.values(), nid_map.keys()))   
# read in cost arrays
with open('cost_arrays.pkl', 'rb') as inp:
    cost_arrays = pickle.load(inp)
with open('td_link_tt.pkl', 'rb') as inp:
    td_link_tt = pickle.load(inp)
# read td_node_cost array
with open('td_node_cost.pkl', 'rb') as inp:
    td_node_cost = pickle.load(inp)
# read link_id_map
with open('link_id_map.pkl', 'rb') as inp:
    link_id_map = pickle.load(inp)
linkID_array = np.array(list(link_id_map.values())).reshape((-1,1))
 #%% **Find TDSP for for different values of b_tt for all times between 8-9am**     
cost_array_dict = {'td_link_tt': td_link_tt, 'td_node_cost':td_node_cost}

b_disc = 0    # a config of b_disc = 1 and b_risk = 0.1 and b_rel=20 has pt/zip combo for Hazelwood-MellonPark
b_risk = 0.10  # 50 cents for every predicted crash in 2 year
b_rel = 15/3600
b_price = 1 # do not adjust

all_paths = {} # form is {(beta_param_1, beta_param_2...): link_sequence}
tdsp_dict = {}
for b_tt in np.arange(0/3600, 22/3600, 2/3600): 
    # final link cost array (generalized travel cost) is a linear combination of the individual components
    cost_final = (b_tt * cost_arrays['travel_time'] + b_disc* cost_arrays['discomfort'] + b_price * cost_arrays['price'] 
                    + b_rel * cost_arrays['reliability'] + b_risk * cost_arrays['risk'])

    # Prepare additional files for compatiblity with MAC-POSTS
    # 4) create link cost file td_link_cost
    filename = 'td_link_cost'
    td_link_cost = np.hstack((linkID_array, cost_final)).astype('float32')
    cost_array_dict['td_link_cost'] = td_link_cost
    # run tdsp from macposts
    macposts_folder = os.path.join(cwd, 'macposts_files')    
    tdsp_dept_time_dict = tdsp_macposts_all_times(macposts_folder, cost_array_dict, inv_nid_map, 30) # {timestamp:tdsp_node_seq}
    tdsp_dict[b_tt] = tdsp_dept_time_dict

# %%
#%% get path info
all_paths = {}
for b_tt, tdsp_dept_time_dict in tdsp_dict.items():
    print(b_tt)
    for t, tdsp_array in tdsp_dept_time_dict.items():
        print(t)
        node_seq = tdsp_array[:,0]  
        link_seq = tdsp_array[:-1,1] 
        path = [nid_map[int(n)] for n in node_seq]
        gtc = tdsp_array[0,2]
        print(path)
        all_paths[(round(b_tt,5),t)] = (node_seq, link_seq, path, gtc)
# #%%
# path_costs = {}
# for (b_tt,timestamp), (node_seq, link_seq, path, gtc) in all_paths.items():
#     path_costs.append((b_tt,timestamp,path,gtc))

# mod_path_attr = list(zip(*path_attr))

#%% now: fix b_tt but adjust t
b_tt = round(12/3600,5)
interval_spacing = 10 # seconds
start_interval = int(30*60/interval_spacing)  # 7:30am
end_interval = int(105*60/interval_spacing)  # 8:45am
path_costs_time = dict(((b_tt,timestamp), all_paths[(b_tt,timestamp)][3]) for timestamp in range(start_interval,end_interval,6))
#cost_dicts = list(path_costs_time.values())
#betas_used = list(zip(*list(path_costs_time.keys()))) # tuple(zip(*list(zip(*path_costs))[0]))
#betas_tt = betas_used[0]
#betas_rel = betas_used[1]

# get which modes define the path
path_modes = {}
for t in range(start_interval,end_interval,6):
    path = all_paths[(b_tt,t)][2]
    stripped_path = [n[0] for n in path]
    mode_count = {}
    for m in ['t','r','b','s']:  # tnc, route, bikeshare, scooter, zip
        mode_count[m] = stripped_path.count(m)
    modes_in_path = ''
    for m, count in mode_count.items():
        if count > 3:
            modes_in_path += m + ','
            path_modes[t] = modes_in_path
#%%
# t = 0
# total_cost_by_time = {}
# for cd in cost_dicts:
#     if cd['price_total'] == 5.50:
#         cd['price_total'] = 2.75  # this is a quick hack that should be resolved in node_cost section
#     total_cost = b_price*cd['price_total'] + b_tt*cd['tt_total'] + b_risk*cd['risk_total'] + b_disc*cd['discomfort_total'] + b_rel*cd['rel_total']
#     total_cost_by_time[t] = total_cost
#     t+=1

df_entries = list(zip(list(range(0,len(path_costs_time))),list(path_costs_time.values()),list(path_modes.values())))
df = pd.DataFrame(df_entries, columns = ['minute', 'total_cost', 'modes_used_list'])
mode_abbrev_dict = {'b,':'bike share + walk', 'r,':'public transit + walk', 'r,b,':'public transit + bikeshare + walk',
                    'r,s,':'scooter + public transit + walk', 't,':'TNC + walk'}
df['modes used'] = df['modes_used_list'].map(mode_abbrev_dict)
df['minute'] = df['minute'].astype('int') + 30  # reset minute 0 to be 7:30
df['hour'] = df['minute'].apply(lambda x: '07' if int(x/60) < 1 else '08')
df['min'] = df['minute'].apply(lambda x: x if x < 60 else x % 60).astype('str').str.zfill(2)
df['hour_min'] = df['hour'] + df['min']
# df['minute'] = '08' + df['minute'].astype('str').str.zfill(2) #'%H%M'
df['departure_time'] = pd.to_datetime(df['hour_min'], format='%H%M').dt.strftime('%H:%M')
fig,ax = plt.subplots(figsize=(16,8))
sns.scatterplot(ax=ax, x='departure_time', y='total_cost', data=df, hue='modes used', s=50)
ax.set_ylabel('generalized travel cost ($)')
ax.set_xlabel('departure time (A.M.)')
ax.tick_params(axis='x', rotation=90)
ax.set_yticks(np.arange(22,26,0.5))
sns.move_legend(ax, 'lower right')
# %%


#%%
# time_start = 7*3600 # seconds start after midnight (7am)
# time_end = 9*3600 # seconds end after midnight (9am)
# interval_spacing = 10 # seconds
# num_intervals = int((time_end-time_start) / interval_spacing)

# new_start = int(num_intervals/2)  # start at 8am (halfway between 7 and 9am)
# intervals_keep = [i for i in range(new_start, num_intervals)]  # keep only results from 8-9am
# new_td_link_tt = td_link_tt.copy()[:,[0]+intervals_keep]


path_costs = {} # this will be a dict of dicts
for (b_tt,timestamp), (node_seq, link_seq, path, gtc) in all_paths.items():
    #print(b_tt,timestamp,path)
    price_total, risk_total, rel_total, tt_total, discomfort_total = 0, 0, 0, 0, 0
    cost_total = 0
    t = int(timestamp)
    for idx, l in enumerate(link_seq):  # the link seq
        # look up how many time intervals it takes to cross the link
        l = int(l) 
        # adjustment of time, if we have reached the last interval
        if t >= td_link_tt.shape[1] - 1:
            t = int(td_link_tt.shape[1] - 1 - 1)
        intervals_cross = td_link_tt[l, t + 1]  # add one bc first col is linkID
        #node_in, node_out = inv_link_id_map[l]   
        # get the price, risk, and reliability of the link at timestamp t
        price_link, risk_link, rel_link, tt_link = cost_arrays['price'][l,t], cost_arrays['risk'][l,t], cost_arrays['reliability'][l,t], cost_arrays['travel_time'][l,t]  # (these arrays do not have a col for linkID)
        discomfort_link = cost_arrays['discomfort'][l,t]
        #print(nid_map[node_in], nid_map[node_out], round(risk_link/60,3))
        #cost_link = td_link_cost[l,t+1]  # cannot use td_link_cost b/c it only reflects most recently used betas     
        # update time and cost totals
        #print(nid_map[node_in], nid_map[node_out], 'price:', round(price_link,2), 'risk:', round(risk_link,2), 'rel:', round(rel_link/60,2), 'tt:',round(tt_link/60,2), 'discomf:',round(discomfort_link,2))
        price_total += (price_link) #+ nodecost)
        risk_total += risk_link
        rel_total += rel_link
        tt_total += tt_link
        discomfort_total += discomfort_link
        #cost_total += cost_link
        cost_attributes = {'price_total': price_total, 'risk_total': risk_total, 'rel_total':rel_total/60, 
                            'tt_total':tt_total/60, 'discomfort_total': discomfort_total}
        t = int(t + intervals_cross-1)  
        #print(nid_map[node_in], nid_map[node_out],round(rel_total/60,2), round(tt_total/60,2))
    path_costs[(round(b_tt,5),timestamp)] = cost_attributes

#%%
# choose a specific timestamp
timestamp = 360
path_costs_time = dict(((round(b_tt,5),timestamp), path_costs[(round(b_tt,5),timestamp)]) for b_tt in np.arange(0/3600, 22/3600, 2/3600))

betas_used = list(zip(*list(path_costs_time.keys()))) # tuple(zip(*list(zip(*path_costs))[0]))
betas_tt = betas_used[0]
#betas_rel = betas_used[1]
cost_dict = list(path_costs_time.values())

all_prices = [d['price_total'] for d in cost_dict]
all_risk = [d['risk_total'] for d in cost_dict]
all_rel = [d['rel_total'] for d in cost_dict]
all_tt = [d['tt_total'] for d in cost_dict]
all_disc = [d['discomfort_total'] for d in cost_dict]

x_axis = np.array(betas_tt)*3600
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(x_axis, all_rel, label='reliability', color='C4', marker='o')
ax1.plot(x_axis, all_tt, label='travel time', color='C2', marker='o')
ax2.plot(x_axis, all_prices, label='price', color='C0', marker='o')
ax2.plot(x_axis, all_risk, label='risk', color='C1', marker='o')
ax2.plot(x_axis, all_disc, label='discomfort', color='C3', marker='o')

ax1.set_zorder(2)
ax2.set_zorder(1)
ax1.patch.set_visible(False)
ax1.axvspan(8, 10, alpha=0.5, color='lightgray')
ax1.axvspan(12, 14, alpha=0.5, color='lightgray')

# distinguish region by mode type
h1 = 25
h2 = 23
ax2.text(1.3,h1,'Public Transit &',fontsize='medium', zorder=3)
ax2.text(2.3,h2,'Walking',fontsize='medium', zorder=3)
ax2.text(9.6,h1,'Bikeshare &',fontsize='medium', zorder=3)
ax2.text(10, h2,'Walking',fontsize='medium', zorder=3)
ax2.text(16.2,h1,'TNC &',fontsize='medium', zorder=3)
ax2.text(15.9,h2,'Walking',fontsize='medium', zorder=3)
ax1.set_ylabel('Travel Time (min), Reliability (min)')
ax2.set_ylabel('Price ($), Risk, Discomfort')
ax1.set_xlabel(r'$\beta_{TT}\ (\$/$hour)')   #$\beta_{TT}$')
ax2.legend(loc='upper right')
ax1.legend(loc='upper left')
ax2.set_xticks(np.arange(0,22,2), fontsize=10)
ax1.set_yticks(np.arange(0,130,20), fontsize=6)
ax2.set_yticks(np.arange(0,50,5), fonrtsize=6)


 #%% **THIS BEGINS THE SENSITIVITY ANALYSIS on scooter pricing**    
import gc
cost_array_dict = {'td_link_tt': td_link_tt, 'td_node_cost':td_node_cost}

b_disc = 0   
b_risk = 0.10  # 50 cents for every predicted crash in 2 year period
b_rel = 15/3600
b_price = 1 # do not adjust
b_tt = 10/3600

all_paths = {} # form is {(beta_param_1, beta_param_2...): link_sequence}
tdsp_dict_sc_price = {}

sc_links = [lid for link, lid in link_id_map.items() if nid_map[link[0]].startswith('sc') and nid_map[link[1]].startswith('sc')]
sc_od_cnx = [lid for link, lid in link_id_map.items() if nid_map[link[0]] == 'org' and nid_map[link[1]].startswith('sc')][0]

price_array_by_scppmin = {}
other_cost_components = b_tt*cost_arrays['travel_time'] + b_disc* cost_arrays['discomfort'] + b_rel*cost_arrays['reliability'] + b_risk*cost_arrays['risk']

for sc_ppmin in 0.01*np.arange(5,40,10):  # change back to 41
    # reduce the scooter cost by x percent
    price_reduction_pct = (0.39 - sc_ppmin) / 0.39  # percent reduction in scoot link cost
    cost_array_price = cost_arrays['price'].copy()
    cost_array_price[sc_links] = (1-price_reduction_pct) * cost_array_price[sc_links]     
    # also remove the $1 usage fee
    cost_array_price[sc_od_cnx] = cost_array_price[sc_od_cnx] - 1  # this is flexible, could make 0.5 for instance 
    price_array_by_scppmin[sc_ppmin] = cost_array_price  # store for future use

    cost_final = (b_price * cost_array_price + other_cost_components)
    # Prepare additional files for compatiblity with MAC-POSTS
    # 4) create link cost file td_link_cost
    filename = 'td_link_cost'
    td_link_cost = np.hstack((linkID_array, cost_final)).astype('float32')
    cost_array_dict['td_link_cost'] = td_link_cost
    # run tdsp from macposts
    macposts_folder = os.path.join(cwd, 'macposts_files')    
    tdsp_dept_time_dict = tdsp_macposts_all_times(macposts_folder, cost_array_dict, inv_nid_map, 30) # {timestamp:tdsp_node_seq}
    tdsp_dict_sc_price[sc_ppmin] = tdsp_dept_time_dict
    # release the memory associated with large link cost array
    del cost_final
    del td_link_cost
    del tdsp_dept_time_dict
    gc.collect()

#%% get path info
all_paths_sc_price = {}
for sc_price, tdsp_dept_time_dict in tdsp_dict_sc_price.items():
    print(sc_price)
    for t, tdsp_array in tdsp_dept_time_dict.items():
        print(t)
        node_seq = tdsp_array[:,0]  
        link_seq = tdsp_array[:-1,1] 
        gtc = tdsp_array[0,2]
        path = [nid_map[int(n)] for n in node_seq]
        print(path)
        all_paths_sc_price[(sc_price,t)] = (node_seq, link_seq, path, gtc)

#%% get which modes define the path
interval_spacing = 10 # seconds
start_interval = int(30*60/interval_spacing)  # 8am
end_interval = int(90*60/interval_spacing)  # 8:30am
path_modes_sc = {}
for sc_ppmin in 0.01*np.arange(5,40,10):
    for t in range(start_interval, end_interval, 6):
        path = all_paths_sc_price[(sc_ppmin,t)][2]
        flag = True if path.count('bsd30') > 1 else False
        stripped_path = [n[0] for n in path]
        mode_count = {}
        for m in ['t','r','b','s']:  # tnc, route, bikeshare, scooter, zip
            mode_count[m] = stripped_path.count(m)
        modes_in_path = ''
        for m, count in mode_count.items():
            if count > 3:
                modes_in_path += m + ','
                path_modes_sc[(sc_ppmin, t)] = (modes_in_path, flag)
        #print(path_modes_sc, '\n')

#%% #[round(sc_ppmin,2), t, 
path_costs_scoot = [[round(sc_ppmin,2), t, 
                    all_paths_sc_price[(sc_ppmin,t)][3],  
                    path_modes_sc[(sc_ppmin, t)][0], 
                    path_modes_sc[(sc_ppmin, t)][1]] 
                    for sc_ppmin in 0.01*np.arange(5,40,10) for t in range(start_interval,end_interval,6)]
#df_entries = list(zip(list(range(0,len(path_costs_time))),list(path_costs_time.values()),list(path_modes.values())))
df = pd.DataFrame(path_costs_scoot, columns = ['sc_ppmin', 'time_interval', 'gtc', 'modes_used', 'flag'])
mode_abbrev_dict = {'b,':'bike share + walk', 'r,':'public transit + walk', 'r,b,':'public transit + bikeshare + walk',
                    'r,s,':'scooter + public transit + walk', 't,':'TNC + walk', 's,':'scooter + walk'}
df.loc[df['flag'] == True,'modes_used']['modes_used'] = ['r'] 
df['modes used'] = df['modes_used'].map(mode_abbrev_dict)
df['minute'] = df['time_interval'].apply(lambda x: int(x*interval_spacing/60) )
df['hour'] = df['minute'].apply(lambda x: '07' if int(x/60) < 1 else '08')
df['min'] = df['minute'].apply(lambda x: x if x < 60 else x % 60).astype('str').str.zfill(2)
df['hour_min'] = df['hour'] + df['min']
# df['minute'] = '08' + df['minute'].astype('str').str.zfill(2) #'%H%M'
df['departure_time'] = pd.to_datetime(df['hour_min'], format='%H%M').dt.strftime('%H:%M')

nrows=2
ncols=2
fig, axs = plt.subplots(figsize=(16,8), sharey=True, nrows=nrows, ncols=ncols)
for i, sc_ppmin in enumerate(0.01*np.arange(5,40,10)):
    row = 0 if i < 2 else 1
    col = i % ncols
    #plt.subplot(a, b, c)
    df_subset = df[df['sc_ppmin'] == round(sc_ppmin,2)]
    sns.scatterplot(ax=axs[row,col], x='departure_time', y='gtc', data=df_subset, hue='modes used', s=50)
    # if i == 0:
    #     plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    # else: 
    #     axs[row,col].get_legend().remove()
    axs[row,col].set_ylabel('generalized travel cost ($)')
    axs[row,col].set_xlabel('departure time (A.M.)')
    axs[row,col].tick_params(axis='x', rotation=90)
    axs[row,col].set_yticks(np.arange(20,26,0.5))
    #sns.move_legend(ax, 'lower right')
    c = c + 1
#%%

path_costs_sc_price = {} # this will be a dict of dicts
for (sc_price,timestamp), (node_seq, link_seq, path, gtc) in all_paths_sc_price.items():
    #print(b_tt,timestamp,path)
    price_total, risk_total, rel_total, tt_total, discomfort_total = 0, 0, 0, 0, 0
    cost_total = 0
    t = int(timestamp)
    for idx, l in enumerate(link_seq):  # the link seq
        # look up how many time intervals it takes to cross the link
        l = int(l) 
        # adjustment of time, if we have reached the last interval
        if t >= td_link_tt.shape[1] - 1:
            t = int(td_link_tt.shape[1] - 1 - 1)
        intervals_cross = td_link_tt[l, t + 1]  # add one bc first col is linkID
        #node_in, node_out = inv_link_id_map[l]   
        # get the price, risk, and reliability of the link at timestamp t
        price_link, risk_link, rel_link, tt_link = price_array_by_scppmin[sc_price][l,t], cost_arrays['risk'][l,t], cost_arrays['reliability'][l,t], cost_arrays['travel_time'][l,t]  # (these arrays do not have a col for linkID)
        discomfort_link = cost_arrays['discomfort'][l,t]
        #print(nid_map[node_in], nid_map[node_out], round(risk_link/60,3))
        #cost_link = td_link_cost[l,t+1]  # cannot use td_link_cost b/c it only reflects most recently used betas     
        # update time and cost totals
        #print(nid_map[node_in], nid_map[node_out], 'price:', round(price_link,2), 'risk:', round(risk_link,2), 'rel:', round(rel_link/60,2), 'tt:',round(tt_link/60,2), 'discomf:',round(discomfort_link,2))
        price_total += (price_link) #+ nodecost)
        risk_total += risk_link
        rel_total += rel_link
        tt_total += tt_link
        discomfort_total += discomfort_link
        #cost_total += cost_link
        cost_attributes = {'price_total': price_total, 'risk_total': risk_total, 'rel_total':rel_total/60, 
                            'tt_total':tt_total/60, 'discomfort_total': discomfort_total}
        t = int(t + intervals_cross-1)  
        #print(nid_map[node_in], nid_map[node_out],round(rel_total/60,2), round(tt_total/60,2))
    path_costs_sc_price[(sc_price,timestamp)] = cost_attributes


#%% choose a specific timestamp
timestamp = 270
path_costs_scoot = dict(((sc_ppmin,timestamp), path_costs_sc_price[(sc_ppmin,timestamp)]) for sc_ppmin in 0.01*np.arange(5,16,2))

#betas_rel = betas_used[1]
cost_dict = list(path_costs_scoot.values())

all_prices = [d['price_total'] for d in cost_dict]
all_risk = [d['risk_total'] for d in cost_dict]
all_rel = [d['rel_total'] for d in cost_dict]
all_tt = [d['tt_total'] for d in cost_dict]
all_disc = [d['discomfort_total'] for d in cost_dict]

x_axis = 0.01*np.arange(5,16,2)
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax2.plot(x_axis, all_prices, label='price', color='C0',  zorder=0, marker='o')
ax2.plot(x_axis, all_risk, label='risk', color='C1',  zorder=0, marker='o')
ax1.plot(x_axis, all_rel, label='reliability', color='C4',  zorder=0, marker='o')
ax1.plot(x_axis, all_tt, label='travel time', color='C2',  zorder=0, marker='o')
ax2.plot(x_axis, all_disc, label='discomfort', color='C3',  zorder=0, marker='o')
# plt.axvline(x=6, color='black',linestyle='--',linewidth=2)
# plt.axvline(x=13, color='black',linestyle='--',linewidth=2)
# ax1.axvspan(6, 8, alpha=0.5, color='gray')
# ax1.axvspan(12, 14, alpha=0.5, color='gray')

# # distinguish region by mode type
# ax2.text(0,29,'Public Transit &',fontsize='medium', zorder=1)
# ax2.text(1.2,27,'Walking',fontsize='medium', zorder=1)
# ax2.text(7.8,29,'Bikeshare &',fontsize='medium', zorder=1)
# ax2.text(8.2,27,'Walking',fontsize='medium', zorder=1)
# ax2.text(20,29,'TNC &',fontsize='medium', zorder=1)
# ax2.text(19.5,27,'Walking',fontsize='medium', zorder=1)
ax1.set_ylabel('Travel Time (min), Reliability (min)')
ax2.set_ylabel('Price ($), Risk, Discomfort')
ax1.set_xlabel(r'scooter price (\$/minute)')   #$\beta_{TT}$')
ax2.legend(loc='upper right')
ax1.legend(loc='upper left')
ax1.set_xticks(0.01*np.arange(5,41,2), fontsize=10)
ax1.tick_params(axis='x', labelrotation = 40)
ax1.set_yticks(np.arange(0,80,20), fontsize=6)
ax2.set_yticks(np.arange(0,50,5), fonrtsize=6)


# %% -----------------STOP HERE----------------------------------------
# now: fix b_tt but adjust t
b_tt = round(12/3600,5)
path_costs_time = dict(((b_tt,timestamp), path_costs[(b_tt,timestamp)]) for timestamp in range(0,360,6))
cost_dicts = list(path_costs_time.values())
#betas_used = list(zip(*list(path_costs_time.keys()))) # tuple(zip(*list(zip(*path_costs))[0]))
#betas_tt = betas_used[0]
#betas_rel = betas_used[1]

# get which modes define the path
path_modes = {}
for t in range(0,360,6):
    path = all_paths[(b_tt,t)][2]
    stripped_path = [n[0] for n in path]
    mode_count = {}
    for m in ['t','r','b','s']:  # tnc, route, bikeshare, scooter, zip
        mode_count[m] = stripped_path.count(m)
    modes_in_path = ''
    for m, count in mode_count.items():
        if count > 3:
            modes_in_path += m + ','
            path_modes[t] = modes_in_path

#%%
import matplotlib.pyplot as plt
import pandas as pd
t = 0
total_cost_by_time = {}
for cd in cost_dicts:
    if cd['price_total'] == 5.50:
        cd['price_total'] = 2.75  # this is a quick hack that should be resolved in node_cost section
    total_cost = b_price*cd['price_total'] + b_tt*cd['tt_total'] + b_risk*cd['risk_total'] + b_disc*cd['discomfort_total'] + b_rel*cd['rel_total']
    total_cost_by_time[t] = total_cost
    t+=1

import seaborn as sns
df_entries = list(zip(list(total_cost_by_time.keys()),list(total_cost_by_time.values()),list(path_modes.values())))
df = pd.DataFrame(df_entries, columns = ['minute', 'total_cost', 'modes_used_list'])
mode_abbrev_dict = {'b,':'bike share + walk', 'r,':'public transit + walk', 'r,b,':'public transit + bikeshare + walk',
                    'r,s,':'scooter + public transit + walk', 't,':'TNC + walk'}
df['modes used'] = df['modes_used_list'].map(mode_abbrev_dict)
df['minute'] = '08' + df['minute'].astype('str').str.zfill(2) #'%H%M'
df['departure_time'] = pd.to_datetime(df['minute'], format='%H%M').dt.strftime('%H:%M')
fig,ax = plt.subplots(figsize=(12,8))
sns.scatterplot(ax=ax, x='departure_time', y='total_cost', data=df, hue='modes used')
ax.set_ylabel('generalized travel cost ($)')
ax.set_xlabel('departure time (A.M.)')
ax.tick_params(axis='x', rotation=90)
ax.set_yticks(np.arange(3,9,0.5))
sns.move_legend(ax, 'upper right')
# %%
