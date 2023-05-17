
# description of file:

#%%
# libraries 
import networkx as nx
import pickle
import os
import re
import config as conf
import util_functions as ut
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from od_connector import od_cnx

# compile supernetwork with od-connectors
# function od_cnx takes the supernetwork as input, then output the supernetwork with od connectors
def compile_G_od(path):
    G_od = od_cnx(path,conf.config_data['Supernetwork']['org'],conf.config_data['Supernetwork']['dst'])
    return G_od

def create_td_cost_arrays(G_super_od):
    cwd = os.getcwd()
    time_start = conf.config_data['Time_Intervals']['time_start']*3600 # seconds start after midnight
    time_end = conf.config_data['Time_Intervals']['time_end']*3600 # seconds end after midnight
    num_intervals = int((time_end-time_start) / conf.config_data['Time_Intervals']['interval_spacing']) # + 1)
    interval_spacing = conf.config_data['Time_Intervals']['interval_spacing']  # seconds

    df_G_super_od = nx.to_pandas_edgelist(G_super_od.graph)
    #df_G_super_od['frc'] = df_G_super_od['frc'].astype('int')
    # read in tt_ratio, reliability_ratio, and PT_headway files
    df_tt_ratio_ext = pd.read_csv(os.path.join(cwd,'Data','Output_Data','intraday_travel_time_ratio.csv'))
    df_rel_ratio_ext = pd.read_csv(os.path.join(cwd,'Data','Output_Data','reliability_ratio.csv'))
    col_dtypes = {'route_id':str, 'direction_id':str, 'stop_id':str, 'traveler_arrival_time':np.int64, 'headway':np.int64}
    df_pt_headway = pd.read_csv(os.path.join(cwd,'Data','Output_Data','PT_headway.csv'), dtype = col_dtypes) 

    # data manipulation
    dept_times = df_tt_ratio_ext['sec_after_midnight'].unique()  # how many unique departure times (every 10 sec from time_start to time_end)
    dept_times.sort()
    interval_mapping = dict(zip(dept_times,range(len(dept_times))))  # fix
    df_tt_ratio_ext['interval'] = df_tt_ratio_ext['sec_after_midnight'].map(interval_mapping)
    df_rel_ratio_ext['interval'] = df_rel_ratio_ext['sec_after_midnight'].map(interval_mapping)


    #%% 1) Travel time cost matrix
    # PUBLIC TRANSIT: BOARDING, TRAVERSAL, and ALIGHTING
    # use df_pt_headway as lookup table
    #df_pt = df_G_super_od[df_G_super_od.mode_type.isin(['pt','board','alight'])][['source','target','mode_type','avg_TT_sec']]
    df_boarding = df_G_super_od[df_G_super_od.mode_type == 'board'][['source','target','mode_type']]
    df_boarding[['stop_id','route_id','direction_id','stop_seq']] = df_boarding.copy()['target'].str.split('_',expand=True)
    df_boarding[['rt','stop_id']] = df_boarding.copy()['stop_id'].str.split('rt',expand=True)
    df_boarding['direction_id'] = df_boarding['direction_id'].astype('str')
    df_boarding = df_boarding[['source','target','mode_type','route_id','direction_id','stop_id']]
    df_boarding_headway = df_boarding.merge(df_pt_headway, how='inner', on=['route_id','direction_id','stop_id'])

    arr_times = df_boarding_headway.traveler_arrival_time.unique().tolist()
    interval_mapping = dict(zip(arr_times,range(len(arr_times))))
    df_boarding_headway['interval'] = df_boarding_headway['traveler_arrival_time'].map(interval_mapping)
    df_boarding_headway = df_boarding_headway.pivot(index=['source','target','mode_type'],columns='interval', values='headway').reset_index()
    interval_colnames = [str(i)+'_avg_tt_sec' for i in range(len(arr_times))]
    colnames = ['source','target','mode_type'] + interval_colnames
    df_boarding_headway.columns = colnames
    df_boarding_headway.insert(3, 'length_m', 0)
    #df_boarding_headway[interval_colnames].values()
    #np.mutiply(df_boarding_headway[interval_colnames].values())

    df_pt_trav = df_G_super_od[df_G_super_od.mode_type == 'pt'][['source','target','mode_type','avg_TT_sec']].reset_index(drop=True)
    # assume pt rides on roads with frc = 2
    pt_trav = df_pt_trav['avg_TT_sec'].values.reshape((len(df_pt_trav),1))
    tt_ratio_frc2 = df_tt_ratio_ext[df_tt_ratio_ext['frc']==2]['tt_ratio'].values
    tt_ratio_frc2 = np.reshape(tt_ratio_frc2,(1,len(tt_ratio_frc2)))
    pt_trav_interval = pd.DataFrame(pt_trav*tt_ratio_frc2)
    pt_trav_interval.columns = interval_colnames
    df_pt_trav_intervals = pd.concat([df_pt_trav[['source','target','mode_type']], pt_trav_interval], axis=1)
    pt_avg_spd = 15 # mph
    pt_avg_spd = pt_avg_spd * conf.config_data['Conversion_Factors']['meters_in_mile'] / 3600
    df_pt_trav_intervals.insert(3, 'length_m', pt_avg_spd*df_pt_trav_intervals['0_avg_tt_sec'])  # assign a length of 0 for consistency with other dfs

    df_alight = df_G_super_od[df_G_super_od.mode_type == 'alight'][['source','target','mode_type']].reset_index(drop=True)
    alight_tt = 5   # sec, can be changed if desired
    alight_tt = alight_tt * np.ones((len(df_alight), 1))  
    alight_interval = pd.DataFrame(alight_tt*np.ones((1,len(interval_mapping))))
    alight_interval.columns = interval_colnames
    df_alight_intervals = pd.concat([df_alight[['source','target','mode_type']], alight_interval], axis=1).reset_index(drop=True)
    df_alight_intervals.insert(3, 'length_m', 0)
    # finally, concatenate all pt dfs together
    df_pt_intervals = pd.concat([df_boarding_headway, df_pt_trav_intervals, df_alight_intervals])

    # TNC and ZIP. can use same process because both are affected by same traffic
    interval_mapping = dict(zip(dept_times,range(len(dept_times))))  # TODO: remove, eventually
    df_tz = df_G_super_od[df_G_super_od.mode_type.isin(['z','t','park'])][['source','target','mode_type','length_m','speed_lim','pred_crash','frc']]
    # first add necessary data for connection edges zd-z and parking edges
    cnx_and_park_edges = ((df_tz.source.str.startswith('zd')) | (df_tz.mode_type == 'park'))
    df_tz.loc[cnx_and_park_edges, 'frc'] = 4   # assume frc=4
    df_tz.loc[cnx_and_park_edges, 'speed_lim'] = 5 # miles per hour

    df_tz = df_tz.sort_values(by='frc').reset_index(drop=True)
    df_tz['frc'] = df_tz['frc'].astype('int')
    df_tz['avg_tt_sec'] = df_tz['length_m'] / (df_tz['speed_lim'] * conf.config_data['Conversion_Factors']['meters_in_mile'] / 3600)

    # separate calculation by frc
    interval_colnames = [str(i)+'_avg_tt_sec' for i in interval_mapping.values()]
    df_tz_intervals = pd.DataFrame()
    df_tz_frc_list = []
    for frc in [2,3,4]:
        df_tz_frc = df_tz[df_tz['frc'] == frc]
        tt_all = df_tz_frc['avg_tt_sec'].values.reshape((len(df_tz_frc),1))
        tt_ratio_frc = df_tt_ratio_ext[df_tt_ratio_ext['frc'] == frc]['tt_ratio'].values
        tt_ratio_frc = np.reshape(tt_ratio_frc,(1,len(tt_ratio_frc)))
        df_tz_frc_list.append(pd.DataFrame(tt_all*tt_ratio_frc))    # matrix multiplication (col of tt * row of tt ratio by interval)
    df_tz_intervals = pd.concat(df_tz_frc_list, ignore_index=True)
    df_tz_intervals.columns = interval_colnames
    df_tz_intervals = pd.concat([df_tz[['source','target','mode_type', 'length_m', 'pred_crash','frc']], df_tz_intervals], axis=1).reset_index(drop=True)

    # tnc waiting mode
    df_twait = df_G_super_od[df_G_super_od.mode_type.isin(['t_wait'])].reset_index(drop=True)
    t_wait_time = conf.config_data['Speed_Params']['TNC']['wait_time'] * 60  # wait time in sec
    df_twait_intervals = pd.DataFrame(t_wait_time * np.ones((len(df_twait), len(interval_colnames))), columns=interval_colnames)
    df_twait_intervals = pd.concat([df_twait[['source','target','mode_type','length_m']],
                                    df_twait_intervals], axis=1)
    df_twait_intervals['length_m'] = df_twait_intervals['length_m'].fillna(0)

    #% ACTIVE MODES: bike share, walk, and scooter: we will do these modes together since the process is the same
    # inherent assumption is that they are not affected by traffic conditions 
    df_active = df_G_super_od[df_G_super_od.mode_type.isin(['bs','sc','w'])][['source','target','mode_type','etype','length_m']].reset_index(drop=True)  # maybe also keep frc
    # adjust euclidean walking distance by a factor of 1.2 (see: circuity factor, levinson)
    df_active.loc[df_active['mode_type'] == 'w', 'length_m'] = 1.2 * df_active.loc[df_active['mode_type'] == 'w', 'length_m']
    speeds = {'bs':conf.config_data['Speed_Params']['bike'], 'sc':conf.config_data['Speed_Params']['scoot'], 'w':conf.config_data['Speed_Params']['walk']}
    df_active['speed'] = df_active['mode_type'].map(speeds)
    df_active['avg_tt_sec'] = df_active['length_m'] / df_active['speed']
    # add an inconvenience cost associated with transferring
    inc = 5 # min
    df_active.loc[df_active.etype=='transfer', 'avg_tt_sec'] =  df_active.loc[df_active.etype=='transfer', 'avg_tt_sec'] + (inc*60)
    tt_all = df_active['avg_tt_sec'].values.reshape(-1,1)  #len()
    bs_sc_intervals = pd.DataFrame(tt_all * np.ones((1,len(interval_mapping))))
    bs_sc_intervals.columns = interval_colnames
    df_active_intervals = pd.concat([df_active[['source','target','mode_type','length_m']], bs_sc_intervals], axis=1).reset_index(drop=True)

    # concatenate all interval dfs to create final travel time cost matrix (df)
    # pd.concat([df_boarding_headway, df_pt_trav_intervals, 
    df_tt_final = pd.concat([df_pt_intervals,
                            df_tz_intervals[['source','target','mode_type','length_m']+interval_colnames],
                            df_twait_intervals, 
                            df_active_intervals[['source','target','mode_type','length_m']+interval_colnames]], 
                            ignore_index=True)

    #%% 2) create reliability cost matrix, following same steps as above
    rel_colnames = [str(i)+'_reliability' for i in interval_mapping.values()]
    #  PUBLIC TRANSIT: BOARDING, TRAVERSAL, and ALIGHTING
    # for boarding use 2*avg. cite: Daryn's paper
    waiting_rel_mult = conf.config_data['Reliability_Params']['board']
    df_boarding_rel_intervals = pd.concat([df_boarding_headway[['source','target','mode_type']], 
                                        waiting_rel_mult * df_boarding_headway[interval_colnames]], axis=1)
    df_boarding_rel_intervals.columns = ['source','target','mode_type'] + rel_colnames
    df_boarding_rel_intervals[rel_colnames] = df_boarding_rel_intervals[rel_colnames].astype('float')

    # for traversal use rel associated with frc=2, consistent with above 
    rel_ratio_frc2 = df_rel_ratio_ext[df_rel_ratio_ext['frc']==2]['rel_ratio'].values.reshape(1,-1)
    df_pt_trav_rel_intervals = pd.DataFrame(df_pt_trav_intervals[interval_colnames].values * rel_ratio_frc2, columns=rel_colnames)
    df_pt_trav_rel_intervals = pd.concat([df_pt_trav_intervals[['source','target','mode_type']],
                                        df_pt_trav_rel_intervals], axis=1)
    # for alighting use 1*avg
    df_alight_rel_intervals = df_alight_intervals.copy().drop(columns='length_m')
    df_alight_rel_intervals.columns = ['source','target','mode_type'] + rel_colnames
    # finally concatenate all dfs together
    df_pt_rel_intervals = pd.concat([df_boarding_rel_intervals,df_pt_trav_rel_intervals, df_alight_rel_intervals])

    # TNC and ZIP
    df_tz.sort_values(by=['frc','source','target'], inplace=True)
    dfs_tz_frc_list = []
    for frc in [2,3,4]:
        df_tz_frc = df_tz_intervals[df_tz_intervals['frc'] == frc]
        rel_ratio_intervals = df_rel_ratio_ext[df_rel_ratio_ext['frc'] == frc]['rel_ratio'].values.reshape((1,len(interval_mapping)))
        rel_ratio_intervals = np.repeat(rel_ratio_intervals,len(df_tz_frc),axis=0)
        tt_frc_arr = df_tz_frc[interval_colnames].values
        dfs_tz_frc_list.append(pd.DataFrame(np.multiply(tt_frc_arr, rel_ratio_intervals), columns=rel_colnames).reset_index(drop=True))
    df_rel_tz_intervals = pd.concat(dfs_tz_frc_list, ignore_index=True)
    df_rel_tz_intervals = pd.concat([df_tz[['source','target','mode_type','length_m']], df_rel_tz_intervals], axis=1, ignore_index=True)
    df_rel_tz_intervals.columns = ['source','target','mode_type','length_m']+rel_colnames

    twait_rel_mult = conf.config_data['Reliability_Params']['t_wait']
    df_twait_rel_intervals = pd.DataFrame(twait_rel_mult * t_wait_time * np.ones((len(df_twait), len(interval_colnames))), columns=rel_colnames)
    df_twait_rel_intervals = pd.concat([df_twait[['source','target','mode_type','length_m']],
                                        df_twait_rel_intervals], axis=1)
    df_twait_rel_intervals['length_m'] = df_twait_rel_intervals['length_m'].fillna(0)

    # ACTIVE MODES: bike share, walk and scooter
    # First, handle the case of transfers to scooters. recall in data_gen_functions.py, we simulated 95th percentile transfer edge length
    to_scoot_tx = (df_G_super_od['mode_type'] == 'w') & (df_G_super_od['target'].str.startswith('sc'))
    df_walk2scoot = df_G_super_od[to_scoot_tx][['source','target','mode_type', 'length_m','95_length_m']].reset_index(drop=True)
    df_walk2scoot.loc[:,'95_tt'] = df_walk2scoot['95_length_m'] / conf.config_data['Speed_Params']['walk']
    df_walk2scoot_intervals = pd.DataFrame(np.repeat(df_walk2scoot['95_tt'].values.reshape((len(df_walk2scoot),1)), len(interval_mapping), axis=1))
    df_walk2scoot_intervals = pd.concat([df_walk2scoot[['source','target','mode_type','length_m']], df_walk2scoot_intervals], axis=1)
    df_walk2scoot_intervals.columns = ['source','target','mode_type','length_m']+rel_colnames
    # Next, handle all other active modes. We will assume that 95% is same as average for active modes (b/sc/w)
    to_scoot_tx = ((df_active_intervals['mode_type'] == 'w') & (df_active_intervals['target'].str.startswith('sc')))
    df_rel_active_intervals = df_active_intervals.copy()[~to_scoot_tx]  # remove scoot tx
    df_rel_active_intervals.columns = ['source','target','mode_type','length_m'] + rel_colnames
    df_rel_active_intervals = pd.concat([df_rel_active_intervals, df_walk2scoot_intervals])

    # combine all reliability dfs together
    df_rel_final = pd.concat([df_pt_rel_intervals,
                            df_rel_tz_intervals.drop(columns=['length_m']), # remove length_m
                            df_twait_rel_intervals.drop(columns=['length_m']),  # remove length_m
                            df_rel_active_intervals.drop(columns=['length_m'])], ignore_index=True)


    #%% 3) PRICE: add separately by mode
    price_params = conf.config_data['Price_Params']
    price_cols = [str(i)+'_price' for i in interval_mapping.values()]
    df_prices_list = []
    for m in df_tt_final.mode_type.unique(): 
        ppmin = price_params[m]['ppmin']
        ppmile = price_params[m]['ppmile']
        fixed_price = price_params[m]['fixed']
        df_m =  df_tt_final[df_tt_final['mode_type'] == m].reset_index(drop=True)  # df associated with the mode m
        length_arr = df_m['length_m'].values.reshape(-1,1)   # for length based price
        length_arr = np.repeat(length_arr, len(interval_colnames), axis=1) / conf.config_data['Conversion_Factors']['meters_in_mile']
        tt_arr = df_m[interval_colnames].values  # for usage based price
        df_prices_m = pd.DataFrame(tt_arr/60 * ppmin + length_arr * ppmile + fixed_price)  # time*price_per_min + length*price_per_mile + fixed_cost
        df_prices_m.columns = price_cols
        df_prices_m = pd.concat([df_m[['source','target','mode_type']], df_prices_m], axis=1)
        # special case: scooter transfer
        if m == 'w':
            # replace the scooter transfer price with $1 to embed fixed cost of scooter into transfer
            sc_tx = (df_prices_m['mode_type'] == 'w') & (df_prices_m['target'].str.startswith('sc'))
            df_prices_m.loc[sc_tx, price_cols] = conf.config_data['Price_Params']['sc_tx']['fixed']
        df_prices_list.append(df_prices_m)
    df_price_final = pd.concat(df_prices_list, ignore_index=True)
    df_price_final.columns = ['source','target','mode_type'] + price_cols

    #%% 4) RISK
    # First fill in pred_crash for all modes: t_wait,board,alight --> 0.
    df_G_super_od.loc[df_G_super_od['mode_type'].isin(['t_wait','board','alight']), 'pred_crash'] = 0
    df_G_super_od.loc[df_G_super_od.bikeway_type.isna(), 'bikeway_type'] = 'None' # data cleaning
    # remove 
    #  for walking, use the crash model
    crash_model = sm.load(os.path.join(cwd,'Data','Output_Data',"crash_model.pickle"))
    # establish walk risk. assume: speed_lim = 35 mph. frc = 3
    df_walk =  df_G_super_od.loc[df_G_super_od['mode_type'] == 'w'][['source','target','mode_type','length_m']].reset_index(drop=True)  # df associated with the mode m
    df_walk['length_meters'] = df_walk['length_m'] 
    df_walk['SPEED'] = 35  # mph
    df_walk['frc'] = 3 
    df_walk['const'] = 1
    df_walk.loc[:,'pred_crash'] = crash_model.predict(df_walk)
    df_walk['bikeway_type'] = 'None'
    preds =  crash_model.predict(df_walk)
    df_walk.loc[:, 'pred_crash'] = preds

    dfs_pred_crash = []
    cols_keep = ['source','target','mode_type','length_m','pred_crash','bikeway_type']
    for m in df_G_super_od['mode_type'].unique():
        if m == 'w':
            df_pred_crash = df_walk[cols_keep]
        else:
            df_pred_crash = df_G_super_od[df_G_super_od['mode_type'] == m][cols_keep]
        dfs_pred_crash.append(df_pred_crash)

    df_pred_crash = pd.concat(dfs_pred_crash, ignore_index=True)

    # some data cleaning: remove rows in df_pred_crash that are not in df_tt_final
    # why: in the step df_boarding.merge(df_pt_headway, how='inner'), we lose some boarding edges
    # recall, df_boarding is derived from df_G_super_od
    df_pred_crash.insert(2, 'source_target', list(zip(df_pred_crash.source, df_pred_crash.target)))
    df_rm = df_pred_crash.merge(df_tt_final.drop_duplicates(), on=['source','target'], how='left', indicator=True)
    df_rm = df_rm[df_rm['_merge'] == 'left_only']
    rm_rows = list(zip(df_rm.source,df_rm.target))
    df_pred_crash = df_pred_crash[~df_pred_crash['source_target'].isin(rm_rows)].drop(columns='source_target').reset_index(drop=True)

    # # get the risk associated with trails (no longer necessay, as we removed the tralis)
    # #TODO: go back to process_street_centerlines.py and fix the pred crash for bikeway_type = 'Trails'
    # # i.e. clean this up by move this part to prior sections
    # trail_rows = df_pred_crash['bikeway_type']=='Trails'
    # df_trails = df_pred_crash[trail_rows]
    # df_trails.loc[:,'length_meters'] = df_trails['length_m']
    # df_trails.loc[:,'SPEED'] = 0  # mph
    # df_trails.loc[:,'frc'] = 4 
    # df_pred_crash.loc[trail_rows, 'pred_crash'] = crash_model.predict(df_trails)

    # Inflate the crash rate by index (cite NHTS)
    crash_idx = conf.config_data['Risk_Crash_Idx']
    # For CMFs, see: Addressing Bicyclist Safety through the Dev of CMFs for Bikeways, pg 126
    df_G_super_od.loc[df_G_super_od.bikeway_type.isna(), 'bikeway_type'] = 'None'
    CMF = {'None':1, 'On Street Bike Route':1, 'Bike Lanes':0.554, 'Cautionary Bike Route':1,
        'cnx':1, 'Protected Bike Lane':0.425, 'Bikeable_Sidewalks':0.554, 'Trails':0.425}
    df_pred_crash['adj_pred_crash'] = df_pred_crash.apply(lambda x: x['pred_crash']*crash_idx[x['mode_type']], axis=1)
    # Final step: adjust for crash modification factor, specifically for micro modes bike and scooter
    micro_modes = ['bs','sc']
    df_pred_crash.loc[df_pred_crash['mode_type'].isin(micro_modes), 'adj_pred_crash'] = df_pred_crash.apply(lambda x: x['adj_pred_crash']*CMF[x['bikeway_type']], axis=1)
    # Create the final risk df
    risk_cols = price_cols = [str(i)+'_risk' for i in interval_mapping.values()]
    adj_pred_crash = df_pred_crash['adj_pred_crash'].values.reshape(-1,1)
    adj_pred_crash_intervals = pd.DataFrame(adj_pred_crash * np.ones((1,len(interval_mapping))), columns=risk_cols)
    df_risk_final = pd.concat([df_pred_crash[['source','target','mode_type']], adj_pred_crash_intervals], axis=1) 

    #%% 5) DISCOMFORT
    # define: segment-level discomfort = segment_discomfort_idx * segment_length (km)
    disc_idxs = conf.config_data['Discomfort_Params']
    disc_columns = [str(i)+'_discomfort' for i in interval_mapping.values()]
    df_disc = df_tt_final.copy()
    df_disc['discomfort'] = df_disc.apply(lambda x: x['length_m']/1000*disc_idxs[x['mode_type']], axis=1)
    df_disc = df_disc[['source','target','mode_type','discomfort']]
    # now repeat discomfort value for all intervals
    d_arr = df_disc['discomfort'].values.reshape(-1,1)
    df_disc_intervals = pd.DataFrame(np.repeat(d_arr, num_intervals, axis=1), columns=disc_columns)
    df_disc_final = pd.concat([df_disc[['source','target','mode_type']], df_disc_intervals], axis=1)

    #%% now that we have all the costs attribute dfs, make sure they are ordered the same way
    df_costs = [df_tt_final, df_rel_final, df_price_final, df_risk_final, df_disc_final]
    for df in df_costs:
        df.sort_values(by=['source','target'], inplace=True)
    # remove length from df_tt_final
    df_tt_final.drop(columns='length_m', inplace=True)
    # store cost dfs in a dict and return from function
    cost_dfs_dict = {'travel_time':df_tt_final, 'reliability':df_rel_final, 'price':df_price_final,
                     'risk':df_risk_final, 'discomfort':df_disc_final}
    return cost_dfs_dict
# now 

#%%
# returns the node cost df where each row has the following form: node_id, in_linkID, out_linkID, cost
def create_node_cost_file(G_super_od, link_id_map):
    inv_nid_map = dict(zip(G_super_od.nid_map.values(), G_super_od.nid_map.keys()))
    #link_id_map = link_id_map #dict(zip(tuple(zip(df_G['source'], df_G['target'])), df_G['linkID']))
    node_costs = []
    for n in list(G_super_od.graph.nodes):
        edges_in = list(G_super_od.graph.in_edges(n))
        edges_out = list(G_super_od.graph.out_edges(n))
        for ei in edges_in:
            ei_num = (inv_nid_map[ei[0]], inv_nid_map[ei[1]])
            for eo in edges_out:            
                eo_num = (inv_nid_map[eo[0]], inv_nid_map[eo[1]])

                # # prevent consecutive transfers (so avoids ps-ps-ps or bs-ps-ps)
                # if ((ut.mode(ei[0]), ut.mode(ei[1])) in G_super_od.pmx) & ((ut.mode(eo[0]), ut.mode(eo[1])) in G_super_od.pmx):
                #     node_costs.append([inv_nid_map[n], link_id_map[ei_num], link_id_map[eo_num], 'double_tx'])
                # # prevent od_cnx - transfer
                # if (ei[0].startswith('org')) & ((ut.mode(eo[0]), ut.mode(eo[1])) in G_super_od.pmx):  
                #     node_costs.append([inv_nid_map[n], link_id_map[ei_num], link_id_map[eo_num], 'double_tx'])
                # # prevent transfer - od_cnx
                # if ((ut.mode(ei[0]), ut.mode(ei[1])) in G_super_od.pmx) & (eo[1].startswith('dst')):  
                #     node_costs.append([inv_nid_map[n], link_id_map[ei_num], link_id_map[eo_num], 'double_tx'])   
                # # alternatively, we can write this in one if statement *i think*: prevent two consecutive walking edges
                # # if G_super_od.graph.edges[ei]['mode_type'] == 'w' & G_super_od.graph.edges[eo]['mode_type'] == 'w' 
                
                # # prevent transfer - tw - t
                # if (G_super_od.graph.edges[ei]['mode_type'] == 'w') & (eo[1].startswith('tw')):
                #     node_costs.append([inv_nid_map[n], link_id_map[ei_num], link_id_map[eo_num], 'double_tx'])

                # account for fee-less PT transfers
                if (n.startswith('ps')) & (G_super_od.graph.edges[ei]['mode_type'] == 'alight') & (eo[1].startswith('ps')) :
                    node_costs.append([inv_nid_map[n], link_id_map[ei_num], link_id_map[eo_num], 'pt_tx'])

                # cannot go backwards i.e. bs1--bsd25--bs1
                if (n.startswith('bs')) & (ei[0] == eo[1]):
                    node_costs.append([inv_nid_map[n], link_id_map[ei_num], link_id_map[eo_num], 'backward'])

    df_nodecost = pd.DataFrame(node_costs, columns = ['node_ID', 'in_link_ID', 'out_link_ID','type'])
    return df_nodecost


# #%%
# # create time-dependent cost dfs for the individual cost components
# # output is a dict of the form: {cost_component: td_cost_df}
# # the cost df has dimensions (num_links x (3 + num_time_intervals)), wheret the 3 add'l columns are due to presence of source, target, mode_type
# # the (i+3)th column is the cost attribute associated with the ith departure interval
# def create_td_cost_arrays(G_super_od):
#     # convert graph to df of link costs
#     df_link = nx.to_pandas_edgelist(G_super_od.graph)
#     cost_factors = ['avg_TT_sec', 'price', 'risk', 'reliability', 'discomfort']
#     # check that lal columns are filled out -- complete
#     # for c in cost_factors:
#     #     col_name = '0_'+c
#     #     #print(df_link[col_name].isna().sum())

#     # Here we build the travel time multiplier as a function of time 
#     # some arbitary linear function
#     len_period = int(conf.config_data['Time_Intervals']['len_period'])
#     num_intervals = int(conf.config_data['Time_Intervals']['len_period'] / conf.config_data['Time_Intervals']['interval_spacing']) + 1
#     print(num_intervals)
#     x = np.linspace(0, len_period, num_intervals )  # x is time [min past] relative to 07:00 AM
#     m = (1.5-1)/len_period # slope
#     y = m*x + 1
#     # plt.plot(x, y, 'o', color='black', zorder=2);
#     # plt.plot(x, y, color='red', zorder=1);
#     # plt.xlabel('Time (seconds relative to 07:00AM)')
#     # plt.ylabel('Travel time multiplier \n (relative to baseline)')

#     cost_factors = ['avg_TT_sec', 'price', 'risk', 'reliability', 'discomfort']
#     cost_factors_0 = ['0_' + c for c in cost_factors]  # _0 represents the 0th departure time interval
#     cols_keep = ['source', 'target', 'mode_type'] + cost_factors_0
#     df_link = df_link[cols_keep]
#     cost_factor_cols = [str(i) +'_' + c for c in cost_factors for i in range(1,num_intervals)]

#     cost_attr_df_dict = {'avg_TT_sec':(), 'price':(), 'reliability':(), 'discomfort':(), 'risk':()}

#     # get all travel time columns representing the TT of the link for different departure time intervals
#     # result is a df with the following cols: source, target, mode_type, 0_avg_TT_sec, 1_avg_TT_sec, 2_avg_TT_sec...
#     # where i_avg_TT_sec represents the avg amount of time it takes to cross the link at the start of the ith departure interval
#     cost_attr_name = 'avg_TT_sec'
#     df_var = df_link[['source','target','mode_type','0_'+cost_attr_name]].copy()
#     interval_cols = [str(n) + '_' + cost_attr_name for n in range(0,num_intervals)]
#     static_tt_modes = ['bs','w','pb','sc','t_wait','alight']   # static modes are those whose travel tiem is independent of traffice (i.e. bike/scooter/walk)
#     dynamic_tt_modes = ['pv','z','t','pt','board']   # dynamic modes are those whose travel time depends on traffic (i.e. vehicle modes)
#     tt_multiplier = y.copy()
#     df_var_all = pd.DataFrame()

#     for m in dynamic_tt_modes:
#         df_m = df_var[df_var['mode_type'] == m].copy()
#         data = [tt_multiplier[idx] * df_m['0_' + cost_attr_name] for idx in range(len(interval_cols))]  # multiply the base TT by the TT muiltiplier
#         data = pd.concat(data, axis=1).set_axis(labels=interval_cols, axis=1)
#         data = pd.concat([df_m[['source','target','mode_type']], data], axis=1)
#         df_var_all = pd.concat([df_var_all, data], axis=0)
#     for m in static_tt_modes:
#         df_m = df_var[df_var['mode_type'] == m].copy()
#         data = [df_m['0_' + cost_attr_name]] *  len(interval_cols)  # repeat the same value 
#         data = pd.concat(data, axis=1).set_axis(labels=interval_cols, axis=1)
#         data = pd.concat([df_m[['source','target','mode_type']], data], axis=1)
#         df_var_all = pd.concat([df_var_all, data], axis=0)
#     cost_attr_df_dict['avg_TT_sec'] = df_var_all.copy() 

#     # get all price columns
#     # result is a df with the following cols: source, target, mode_type, 0_price, 1_price, 2_price...
#     # where i_price represents the price it takes to cross the link at the start of the ith departure interval
#     cost_attr_name = 'price'
#     df_var = df_link[['source','target','mode_type', '0_avg_TT_sec', '0_'+cost_attr_name]].copy()
#     interval_cols = [str(n) + '_' + cost_attr_name for n in range(0,num_intervals)]
#     fixed_price_modes = ['pv','pb', 'pt', 'board', 'alight', 'w','t_wait']    # modes that have a fixed price
#     usage_price_modes = ['z','t','bs','sc']    # modes whose price depends on usage time 
#     usage_prices = conf.config_data['Price_Params']
#     usage_prices = {'z':usage_prices['zip']['ppmin'], 't':usage_prices['TNC']['ppmin'], 'bs':usage_prices['bs']['ppmin'],
#                     'sc':usage_prices['scoot']['ppmin']}
#     df_var_all = pd.DataFrame()

#     for m in usage_price_modes:
#         df_m = df_var[df_var['mode_type'] == m].copy()
#         # TODO: for each unimodal graph, check that '0_price' is just the fixed price 
#         data = [df_m['0_' + 'price'] +   # fixed mileage component (for TNC) + price for usage time
#                     tt_multiplier[idx] * usage_prices[m] / 60 * df_m['0_avg_TT_sec'] for idx in range(len(interval_cols))]
#         data = pd.concat(data, axis=1).set_axis(labels=interval_cols, axis=1)
#         data = pd.concat([df_m[['source','target','mode_type']], data], axis=1)
#         df_var_all = pd.concat([df_var_all, data], axis=0)
#     for m in fixed_price_modes:
#         df_m = df_var[df_var['mode_type'] == m].copy()
#         data = [df_m['0_' + cost_attr_name]] *  len(interval_cols)  # repeat the same value 
#         data = pd.concat(data, axis=1).set_axis(labels=interval_cols, axis=1)
#         data = pd.concat([df_m[['source','target','mode_type']], data], axis=1)
#         df_var_all = pd.concat([df_var_all, data], axis=0)

#     cost_attr_df_dict['price'] = df_var_all.copy()

#     # get all reliability columns
#     modes = df_link['mode_type'].unique().tolist()
#     cost_attr_name = 'reliability'
#     df_var = df_link[['source','target','mode_type', '0_avg_TT_sec', '0_'+cost_attr_name]].copy()
#     interval_cols = [str(n) + '_' + cost_attr_name for n in range(0,num_intervals)]
#     rel_weights = conf.config_data['Reliability_Params']
#     rel_weights = {'pt':rel_weights['PT_traversal'], 'board':rel_weights['PT_wait'], 't':rel_weights['TNC'],
#                     'pb':rel_weights['pb'], 'pv':rel_weights['pv'], 'sc':rel_weights['scoot'], 'bs':rel_weights['bs'],
#                     'z':rel_weights['zip'], 'w':rel_weights['walk'], 't_wait':rel_weights['TNC_wait'],
#                     'sc_wait': 1.25,  # put this in the config file
#                     'alight':1}
#     df_var_all = pd.DataFrame()  

#     # special case: scooter transfers 
#     # we do the soooter transfers separately becasue we already generated 95th percentile TT, hence we will not create it as rel_weight * avgTT as we do below for the other modes
#     df_sc_tx = df_var[(df_var['mode_type'] == 'w') & (df_var['target'].str.startswith('sc'))].copy() # filter by rows going TO scooter
#     data = [df_sc_tx['0_' + cost_attr_name]] *  len(interval_cols)
#     data = pd.concat(data, axis=1).set_axis(labels=interval_cols, axis=1)
#     data = pd.concat([df_sc_tx[['source','target','mode_type']], data], axis=1)
#     df_var_all = pd.concat([df_var_all, data], axis=0)  

#     # note: we are using df_TT because it already has the time cost for each interval
#     df_all_other = cost_attr_df_dict['avg_TT_sec'][~((cost_attr_df_dict['avg_TT_sec']['mode_type'] == 'w') & (cost_attr_df_dict['avg_TT_sec']['target'].str.startswith('sc')))]
#     for m in modes:
#         # reliability is defined as reliability_coef * avg_tt, for all intervals
#         df_m = df_all_other[df_all_other['mode_type'] == m].copy()
#         data = [rel_weights[m] * df_m[str(i) + '_avg_TT_sec'] for i in range(len(interval_cols))]
#         data = pd.concat(data, axis=1).set_axis(labels=interval_cols, axis=1)
#         data = pd.concat([df_m[['source','target','mode_type']], data], axis=1)
#         df_var_all = pd.concat([df_var_all, data], axis=0)

#     cost_attr_df_dict['reliability'] = df_var_all.copy()
#     # TODO: right now, tnc waiting time is constant by departure time. could think about changing it 

#     # add discomfort and risk. assume constant by departure interval
#     for c in ['discomfort', 'risk']:
#         df_var = df_link[['source','target','mode_type','0_' + c]].copy()
#         interval_cols = [str(n) + '_' + c for n in range(0,num_intervals)]
#         data = [df_var['0_' + c]] *  len(interval_cols)
#         data = pd.concat(data, axis=1).set_axis(labels=interval_cols, axis=1)
#         data = pd.concat([df_var[['source','target','mode_type']], data], axis=1)
#         cost_attr_df_dict[c] = data

#     for df in cost_attr_df_dict.values():
#         df.sort_values(by=['source','target','mode_type'], inplace=True)
    
#     return cost_attr_df_dict


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

