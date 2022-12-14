#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 17:37:04 2022

@author: lindsaygraff

Build config file
"""

import yaml

# Define parameters
config_info = {
    'Geography': {
        'neighborhoods' : ['Hazelwood', 'Glen Hazel', 'Greenfield', 'Squirrel Hill South', 'Squirrel Hill North'],
                           #'Shadyside'] # Shadyside #, 'Point Breeze North', 'Larimer']
                           
                           # 'Shadyside', 'South Oakland', 'Central Oakland', 'North Oakland', 'Bloomfield', 
                           # 'Friendship', 'Garfield',
                           # 'East Liberty', 'Larimer', 'Point Breeze']
        },
    'Beta_Params': {
        'b_price': 1,
        'b_TT': 10/3600,   # when b_tt and b_rel are 10/60, the route is scoot-zip 
        'b_risk': 0.1,
        'b_disc': 0,
        'b_rel': 10/3600
        },  # ultimately need b_rel_weight * b_TT
    'Speed_Params': {
        'walk': 1.3,  # m/s
        #'parking_speed': 5 / 3600 * 1609, # miles /hr / 3600 s/hr * 1609 meter/mile = m/s
        'scoot': 15 / 3600 * 1000,    # 15 km/hr / (3600 s/hr ) * 1000 m/km = m/s
        'bike': 15 / 3600 * 1000,    # 15 km/hr / (3600 s/hr ) * 1000 m/km = m/s
        'TNC': {'wait_time': 7}  # minutes
        },   # km/hr
    'Price_Params': {
        'walk': {'ppmin': 0},  # fix the scooter price back!
        'scoot': {'ppmin': 0.39, 'fixed': 1},  # $
        'bs': {'ppmin': 25/200},
        'TNC': {'ppmin': 0.18, 'ppmile': 1.08, 'fixed': 2.92 + 1.53 + 1.81, 'minfare_buffer': 8.32/4},  # fixed price is: base fare + "long pickup fare" + "booking fee"
        'PT': {'fixed': 2.75},
        'pb': {'ppmin': 0},
        'zip': {'ppmin': 11/60, 'fixed_per_month': 9, 'est_num_trips': 4},
        'pv': {'ppmin':0, 'ppmile': 0.20}
        # 'walk_ppmin': 0,
        # 'scoot_ppmin': 0.39,
        # 'scoot_fix_price': 1,
        # 'bs_ppmin': 20/300,
        # 'TNC_fix_price': 1.51 + 1.60,
        # 'TNC_ppmile': 0.92,
        # 'TNC_ppmin': 0.34,
        # 'TNC_minfare_buffer': 8.32/4,
        # 'PT_price': 2.75,
        # 'pb_ppmin': 0,
        # 'pv_ppmile': 0.20,
        # 'zip_ppmin': 11/60,
        # 'zip_fixed_ppmonth': 9,
        # 'zip_est_num_trips': 4
        },
    'Conversion_Factors': {
        'meters_in_mile': 1609,
        'miles_in_km': 0.621371
        },
    'Time_Intervals': {
        'interval_spacing': 20,  # sec
        'len_period': 60*60,  # sec
        'time_start': 7,  # AM
        'time_end': 9  # AM
        },
    # 'Active_Mode_Parameters': {
    #     'active_modes': ['w','pb','bs','sc'],
    #     'nonactive_modes': ['pv','pt','t','z'],z
    #     'discomf_weight_bike': 3/10,
    #     'discomf_weight_scoot_walk': 1/10,
    #     'discomf_weight_nonactive': 0,
    #     'rel_weight_active': 1,
    #     'rel_weight_nonactive': 1.5
    #     },
    'Reliability_Params': {
        'walk': 1,
        'scoot': 1,
        'pv': 1.5,
	'pb': 1,
        'bs': 1,
        'PT_wait': 2, 
        'PT_traversal': 1.5,
	'TNC': 1.5,
        'TNC_wait': 2,
	'zip': 1.5,
        'drive': 1.5,
 	'bike': 1
        },
    'Discomfort_Params': {
        'walk': 1.1,
        'scoot': 1.1,
        'pb': 1.3,
        'bs': 1.3,
        'PT_traversal': 1.1,
        'PT_wait': 1.1,  # could change if thinking about cold weather conditions and waiting outside is unpleasant
        'pv': 1,
        'TNC': 1,
        'zip': 1
        },
    'Risk_Parameters': {
        'walk': 1.05,
        'scoot': 1.1,
        'pb': 1.1,
        'bs': 1.1,
        'PT_traversal': 1,
        'PT_wait': 1,  # a nonzero parameter is meant to indicate the risk associated with waiting idly at a bus stop
        'pv': 1,
        'TNC': 1,
        'zip': 1,
        #'risk_weight_active': 1.2,
        'crash_weight': 5
        },
    'Connection_Edge_Speed': {
        'pv': 5 / 3600 * 1609, # miles /hr / 3600 s/hr * 1609 meter/mile = m/s
        'bs': 15 / 3600 * 1000,    # 15 km/hr / (3600 s/hr ) * 1000 m/km = m/s
	'zip': 5 / 3600 * 1609, # miles /hr / 3600 s/hr * 1609 meter/mile = m/s
        },
    'Scoot_Data_Generation': {
        'num_days_of_data': 30,
        'num_obs': 500
        },
    'Supernetwork': {
        'modes_included': ['bs', 'z', 'sc', 't', 'pt'],
        'W_tx': 0.5,  # miles,
        'W_od': 0.75,  # miles
        #'org': [-79.92963073354404, 40.406957045578174] ,  # glen hazel neighborhood
        #'dst': [-79.9275707970418, 40.44563703158395]  # squirrel hill north neighborhood
	    'org': [-79.94868171046522, 40.416379503934145],  # hazelwood green
	    'dst': [-79.91944888168011, 40.45228774674678] # mellon park 
    # chatham univ: [-79.92399900719222, 40.44955761072877] 
        }  #, 'pb']}
    
    }

# Write the config file
with open("config.yaml", 'w') as yamlfile:
    data = yaml.dump(config_info, yamlfile)
    print("Write successful")

