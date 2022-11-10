# libraries
import os
import numpy as np
import pickle
import copy
import pandas as pd
import util_functions as ut
import networkx as nx
import config as conf
import os
import re
import numpy as np
import pickle
import MNMAPI

#os.path.join(cwd, 'Data', 'Output_Data', 'G_super_od.pkl')

def TDSP(graph_filepath, timestamp, beta_price):
    # read pickled supernetwork, complete with transfer edges and od cnx
    cwd = os.getcwd()
    with open(graph_filepath, 'rb') as inp:
        G_super = pickle.load(inp)
    # also adjust the inverse nidmap
    inv_nid_map = dict(zip(G_super.nid_map.values(), G_super.nid_map.keys()))   

    # now convert to link cost file
    df_link = nx.to_pandas_edgelist(G_super.graph)
    # assign total cost as linear combination of all 5 cost factors
    cost_factors = ['avg_TT_min', 'price', 'risk', 'reliability', 'discomfort']
    cost_factor_cols = []
    num_intervals = int(conf.config_data['Time_Intervals']['len_period'] / conf.config_data['Time_Intervals']['interval_spacing']) + 1
    for c in cost_factors:
        for i in range(num_intervals):
            cost_factor_cols.append('interval'+str(i)+'_' + c)
    cols_keep = ['source', 'target', 'mode_type'] + cost_factor_cols
    df_link = df_link[cols_keep]
    betas = conf.config_data['Beta_Params']
    for i in range(num_intervals):
        df_link['interval'+str(i)+'_' + 'cost'] = (betas['b_TT'] * df_link['interval'+str(i)+'_' + 'avg_TT_min']
                                                   + betas['b_disc'] * df_link['interval'+str(i)+'_' + 'discomfort']
                                                   + beta_price * df_link['interval'+str(i)+'_' + 'price']
                                                   + betas['b_rel'] * df_link['interval'+str(i)+'_' + 'reliability']
                                                   #+ beta_risk * df_link['interval'+str(i)+'_' + 'risk'])
                                                   + betas['b_risk'] * df_link['interval'+str(i)+'_' + 'risk'])
    cost_cols = ['interval'+str(i)+'_' + 'cost' for i in range(num_intervals)]
    df_linkcost = df_link[['source','target'] + cost_cols]
    # add link id
    df_linkcost['linkID'] = df_linkcost.index
    print(df_linkcost.shape)
    # then make separate network topology file, called df_G
    df_G = df_linkcost[['linkID', 'source', 'target']]
    print(df_G.shape)

    # map alphanumeric node names to their numeric names
    df_G = df_G.copy()
    df_G['source'] = df_G['source'].map(inv_nid_map)
    df_G['target'] = df_G['target'].map(inv_nid_map)

    # *****************
    # add cost of parking to zipcar:
    # **THIS IS SOMETHING TO MOVE TO THE SECTION OF BUILDING UNIMODAL GRAPHS**
    park_hours = 2

    df_linkcost = df_linkcost.copy()
    for index, row in df_linkcost[df_linkcost['target'].str.startswith('kz')].iterrows():
        parking_rate = G_super.graph.nodes[row['target']]['float_rate']
        # add park_hours * (parking_rate per hour + zip cost per hour) to each cost
        for i in range(num_intervals):
            df_linkcost.at[index, 'interval' + str(i) + '_cost'] += park_hours * (parking_rate + conf.config_data['Price_Params']['zip']['ppmin']*60)
    # *****************

    df_linkcost = df_linkcost.copy()
    df_linkcost['source'] = df_linkcost['source'].map(inv_nid_map)
    df_linkcost['target'] = df_linkcost['target'].map(inv_nid_map)

    #%% Prepare files for compatiblity with MAC-POSTS
    # 1) Create graph topology file
    filename = 'graph'
    np.savetxt(filename, df_G, fmt='%d', delimiter=' ')
    f = open(filename, 'r')
    log = f.readlines()
    f.close()
    log.insert(0, 'EdgeId FromNodeId ToNodeId\n')
    f = open(filename, 'w')
    f.writelines(log)
    f.close()

    # 2) create link cost file 
    filename = 'td_link_cost'
    np.savetxt(filename, df_linkcost[['linkID'] + cost_cols], fmt='%d ' + (num_intervals-1)*'%f ' + '%f')
    f = open(filename, 'r')
    log = f.readlines()
    f.close()
    log.insert(0, 'link_ID td_cost\n')
    f = open(filename, 'w')
    f.writelines(log)
    f.close()

    # 3) # Create node cost df to prevent org-ps-ps or ps-ps-dst transfers
    link_id_map = dict(zip(tuple(zip(df_G['source'], df_G['target'])), df_G['linkID']))
    node_costs = []
    for n in list(G_super.graph.nodes):
        edges_in = list(G_super.graph.in_edges(n))
        edges_out = list(G_super.graph.out_edges(n))
        for ei in edges_in:
            ei_num = (inv_nid_map[ei[0]], inv_nid_map[ei[1]])
            for eo in edges_out:            
                eo_num = (inv_nid_map[eo[0]], inv_nid_map[eo[1]])
                # prevent consecutive transfers (so avoids ps-ps-ps or bs-ps-ps)
                if ((ut.mode(ei[0]), ut.mode(ei[1])) in G_super.pmx) & ((ut.mode(eo[0]), ut.mode(eo[1])) in G_super.pmx):
                    node_costs.append([inv_nid_map[n], link_id_map[ei_num], link_id_map[eo_num]])
                # prevent od_cnx - transfer
                if (ei[0].startswith('org')) & ((ut.mode(eo[0]), ut.mode(eo[1])) in G_super.pmx):  
                    node_costs.append([inv_nid_map[n], link_id_map[ei_num], link_id_map[eo_num]])
                # prevent transfer - od_cnx
                if ((ut.mode(ei[0]), ut.mode(ei[1])) in G_super.pmx) & (eo[1].startswith('dst')):  
                    node_costs.append([inv_nid_map[n], link_id_map[ei_num], link_id_map[eo_num]])   

    df_nodecost = pd.DataFrame(node_costs, columns = ['node_ID', 'in_link_ID', 'out_link_ID'])
    for i in range(num_intervals):
        df_nodecost['interval'+str(i)+'_COST'] = 100000 # some arbitarily large number to prevent unwanted transfers

    filename = 'td_node_cost'
    np.savetxt(filename, df_nodecost, fmt='%d %d %d ' + (num_intervals-1)*'%f ' + '%f')
    f = open(filename, 'r')
    log = f.readlines()
    f.close()
    log.insert(0, 'node_ID in_link_ID out_link_ID td_cost\n')
    f = open(filename, 'w')
    f.writelines(log)
    f.close()

    # 4) edit the config file
    # function to edit the config file for compatibility with MAC-POSTS
    def edit_config(folder, graph_name, len_link_file, len_node_file):
        with open(folder + '/config.conf', 'w') as f:
            f.write('[Network] \n')
            f.write('network_name = ' + graph_name + '\n')
            f.write('num_of_link = ' + str(len_link_file) + '\n')
            f.write('num_of_node = ' + str(len_node_file) + '\n')
            
    folder = os.getcwd()
    edit_config(folder, 'graph', df_linkcost.shape[0], df_nodecost.shape[0])

    # 5) add time-dep travel time for each link and node (just TT, not full cost)
    # travel time
    df_link['source'] = df_link['source'].map(inv_nid_map)
    df_link['target'] = df_link['target'].map(inv_nid_map)
    # add link id
    df_link['linkID'] = df_link.index
    tt_cols = ['interval'+str(i)+'_' + 'avg_TT_min' for i in range(num_intervals)]
    df_tt = df_link[['linkID'] + tt_cols]
    # create td_link_tt
    np.savetxt(os.path.join(folder, 'td_link_tt'), df_tt, fmt='%d ' + (num_intervals-1)*'%f ' + '%f')
    f = open(os.path.join(folder, 'td_link_tt'), 'r')
    log = f.readlines()
    f.close()
    log.insert(0, 'link_ID td_tt\n')
    f = open(os.path.join(folder, 'td_link_tt'), 'w')
    f.writelines(log)
    f.close()

    # create td_node_tt
    td_node_tt = np.zeros((len(df_nodecost), num_intervals))
    node_ID = df_nodecost[['node_ID', 'in_link_ID', 'out_link_ID']].to_numpy()
    td_node_tt = np.concatenate((node_ID, td_node_tt), axis=1)
    np.savetxt(os.path.join(folder, 'td_node_tt'), td_node_tt, fmt='%d %d %d ' + (num_intervals-1)*'%f ' + '%f')
    f = open(os.path.join(folder, 'td_node_tt'), 'r')
    log = f.readlines()
    f.close()
    log.insert(0, 'node_ID in_link_ID out_link_ID td_tt\n')
    f = open(os.path.join(folder, 'td_node_tt'), 'w')
    f.writelines(log)
    f.close()

    # 6) identify params for shortest path
    max_interval = num_intervals
    num_rows_link_file = df_linkcost.shape[0]
    num_rows_node_file = df_nodecost.shape[0]
    link_cost_file_name = "td_link_cost"
    node_cost_file_name = "td_node_cost"
    link_tt_file_name = 'td_link_tt'
    node_tt_file_name = 'td_node_tt'

    dst_node_ID = inv_nid_map['dst']
    org_node_ID = inv_nid_map['org']

    # invoke MAC-POSTS
    tdsp_api = MNMAPI.tdsp_api()

    tdsp_api.initialize(folder, max_interval, num_rows_link_file, num_rows_node_file, 
                        link_tt_file_name, node_tt_file_name, link_cost_file_name, node_cost_file_name)
    tdsp_api.build_tdsp_tree(dst_node_ID)
    timestamp = timestamp
    tdsp = tdsp_api.extract_tdsp(org_node_ID, timestamp)
    path = tdsp[:,0]
    cost = tdsp[0,2]
    TT = tdsp[0,3]
    #np.savetxt(os.path.join(folder,'shortest_path_ex3'), path)
	
    for n in tdsp[:,0]:
        print(G_super.nid_map[int(n)])
    print('cost:', tdsp[0,2])
    print('TT:', tdsp[0,3])

    return(path, cost, TT)

# conduct sensitivity analysis by varying risk parameter over some range 
cwd = os.getcwd()
travel_times = []
for r in np.arange(0,1,0.25):
    path, cost, TT = TDSP(os.path.join(cwd, 'Data', 'Output_Data', 'G_super_od.pkl'), 6, r)
    travel_times.append(TT)
print(travel_times)



