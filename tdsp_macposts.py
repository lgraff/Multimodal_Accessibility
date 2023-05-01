# invoke MAC-POSTS
# libraries
import MNMAPI
import numpy as np
import pickle
import os
import config as conf

cwd = os.getcwd()
#array_path = os.path.join(folder, 'macposts_files', 'macposts_arrays.npz')

# create config file
# function to edit the config file for compatibility with MAC-POSTS
def edit_config(folder, graph_name, num_links, num_nodes):
    with open(folder + '/config.conf', 'w') as f:
        f.write('[Network] \n')
        f.write('network_name = ' + graph_name + '\n')
        f.write('num_of_link = ' + str(num_links) + '\n')
        f.write('num_of_node = ' + str(num_nodes) + '\n')

def tdsp_macposts(macposts_folder, cost_arrays, inv_nid_map, timestamp):
    # load the compressed numpy arrays
    #cost_arrays = np.load(os.path.join(macposts_folder, arrays_filename))
    # for testing, only use 50 time intervals and the first 100 node costs
    time_start = conf.config_data['Time_Intervals']['time_start']*3600 # seconds start after midnight
    time_end = conf.config_data['Time_Intervals']['time_end']*3600 # seconds end after midnight
    interval_spacing = conf.config_data['Time_Intervals']['interval_spacing']
    num_intervals = int((time_end-time_start) / interval_spacing)

    intervals_keep = [i for i in range(timestamp, num_intervals)]
    #a = td_link_tt[:5, [0]+intervals_keep]
    td_link_cost = cost_arrays['td_link_cost'][:, [0]+intervals_keep].copy().astype(float)  # add one for link id
    td_link_tt = cost_arrays['td_link_tt'][:, [0]+intervals_keep].copy().astype(float)
    td_node_cost = cost_arrays['td_node_cost'][:, [0,1,2]+intervals_keep].copy().astype(float)  # add three for in_node, pass-through_node, end_node

    max_interval = td_link_cost.shape[1] - 1  # subtract one for the link_id
    # print(max_interval)
    # print(td_link_cost.shape)
    # print(td_node_cost.shape)

    num_rows_link_file = td_link_cost.shape[0]
    num_rows_node_file = cost_arrays['td_node_cost'].shape[0]
    nodes = list(inv_nid_map.values())
    num_nodes = len(nodes)

    # edit the config file
    edit_config(macposts_folder, 'graph', num_rows_link_file, num_nodes)

    # invoke TDSP api from mac-posts
    tdsp_api = MNMAPI.tdsp_api()
    tdsp_api.initialize(macposts_folder, max_interval, num_rows_link_file, num_rows_node_file)
    tdsp_api.read_td_cost_py(td_link_tt, td_link_cost, td_node_cost, td_node_cost)  # assume td_node_cost = td_node_tt
    print('TDSP api has successfully read the files')

    # find tdsp
    tdsp_api.build_tdsp_tree(inv_nid_map['dst'])
    new_timestamp = 0 #timestamp  because we have now turned the timestamp into the zeroth index
    tdsp = tdsp_api.extract_tdsp(inv_nid_map['org'], new_timestamp)
    
    return(tdsp) #  (tdsp[:,0].tolist(), tdsp[:-1,1].tolist()))  # (node sequence, link sequence)

    # tdsp : number of nodes * 4
    # first col: node sequence
    # second col: link sequence with last element being -1
    # third col: first element being the travel cost
    # fourth col: first element being the travel time
    #print(tdsp)


def tdsp_macposts_all_times(macposts_folder, cost_arrays, inv_nid_map):
    # load the compressed numpy arrays
    #cost_arrays = np.load(os.path.join(macposts_folder, arrays_filename))
    # for testing, only use 50 time intervals and the first 100 node costs
    time_start = conf.config_data['Time_Intervals']['time_start']*3600 # seconds start after midnight
    time_end = conf.config_data['Time_Intervals']['time_end']*3600 # seconds end after midnight
    interval_spacing = conf.config_data['Time_Intervals']['interval_spacing']
    num_intervals = int((time_end-time_start) / interval_spacing)

    timestamp = int(num_intervals/2)
    intervals_keep = [i for i in range(timestamp, num_intervals)]
    #a = td_link_tt[:5, [0]+intervals_keep]
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
    # get the sp every 60 seconds for 30 minutes (from 8am to 8:30am)
    intervals_in_min = int(60/interval_spacing)  # number of time intervals in 1 minute
    new_timestamp = 0  # b/c we have zero-indexed the timestamp
    tdsp_dict = {}
    for i in range(30):
        tdsp_arr = tdsp_api.extract_tdsp(inv_nid_map['org'], new_timestamp)
        tdsp_dict[new_timestamp] = tdsp_arr
        new_timestamp += intervals_in_min
    return(tdsp_dict) 

    # tdsp = tdsp_api.extract_tdsp(inv_nid_map['org'], new_timestamp)
    # return(tdsp)
    


    # nid_map = dict(zip(inv_nid_map.values(), inv_nid_map.keys()))
    # for n in tdsp[:,0]:
    #     print(nid_map[int(n)])
    # print('cost:', tdsp[0,2])
    # print('TT:', tdsp[0,3])

# macposts_folder = os.path.join(cwd, 'macposts_files')
# arrays_filename = 'macposts_arrays.npz'
# # read the inverse node id map: this maps a string node name (such as 'org') to an integer value
# inv_nid_path = os.path.join(cwd, 'macposts_files', 'inv_nid_map.pickle')
# with open(inv_nid_path, 'rb') as handle:
#     inv_nid_map = pickle.load(handle)
# timestamp = 5

# tdsp_macposts(macposts_folder, arrays_filename, inv_nid_map, 5)