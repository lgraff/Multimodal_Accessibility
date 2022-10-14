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
        'neighborhoods' : ['Hazelwood', 'Glen Hazel', 'Greenfield', 'Squirrel Hill South', 'Squirrel Hill North']
                           
                           # 'Shadyside', 'South Oakland', 'Central Oakland', 'North Oakland', 'Bloomfield', 
                           # 'Friendship', 'Garfield',
                           # 'East Liberty', 'Larimer', 'Point Breeze']
        },
    'Beta_Params': {
        'b_price': 1,
        'b_TT': 10/60,
        'b_risk_weight': 1,
        'b_disc_weight': 0.5,
        'b_rel_weight': 0.75
        },  # ultimately need b_rel_weight * b_TT
    'Speed_Params': {
        'walk': 1.3,  # m/s
        #'parking_speed': 5 / 3600 * 1609, # miles /hr / 3600 s/hr * 1609 meter/mile = m/s
        'scoot': 15 / 3600 * 1000,    # 15 km/hr / (3600 s/hr ) * 1000 m/km = m/s
        'bike': 15 / 3600 * 1000,    # 15 km/hr / (3600 s/hr ) * 1000 m/km = m/s
        'TNC': {'wait_time': 7}  # minutes
        },   # km/hr
    'Price_Params': {
        'walk': {'ppmin': 0},
        'scoot': {'ppmin': 0.39, 'fixed': 1},  # $
        'bs': {'ppmin': 20/300},
        'TNC': {'ppmin': 0.34, 'ppmile': 0.92, 'fixed': 1.51 + 1.60, 'minfare_buffer': 8.32/4},
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
        'interval_spacing': 10,  # min
        'len_period': 120,  # min
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
        'drive': 1.5,
        'bike': 1,
        'PT_wait': 2, 
        'PT_traversal': 1.5,
        'TNC_wait': 2
        },
    'Discomfort_Params': {
        'walk': 1/10,
        'scoot': 1/10,
        'pb': 3/10,
        'bs': 3/10,
        'PT_traversal': 0,
        'PT_wait': 0,  # could change if thinking about cold weather conditions and waiting outside is unpleasant
        'pv': 0,
        'TNC': 0,
        'zip': 0,
        'drive': 0,
        'bike': 3/10},
    'Risk_Parameters': {
        'risk_weight_active': 1.2,
        'crash_weight': 5,
        'PT_traversal': 1,
        'PT_wait': 0.5  # a nonzero parameter is meant to indicate the risk associated with waiting idly at a bus stop
        },
    'Connection_Edge_Speed': {
        'drive': 5 / 3600 * 1609, # miles /hr / 3600 s/hr * 1609 meter/mile = m/s
        'bike': 15 / 3600 * 1000    # 15 km/hr / (3600 s/hr ) * 1000 m/km = m/s
        },
    'Scoot_Data_Generation': {
        'num_days_of_data': 30,
        'num_obs': 100
        },
    'Supernetwork': {
        'modes_included': ['bs', 'z', 'sc', 't', 'pt'],
        'W_tx': 0.5,  # miles,
        'W_od': 1.0  # miles
        }  #, 'pb']}
    
    }

# Write the config file
with open("config.yaml", 'w') as yamlfile:
    data = yaml.dump(config_info, yamlfile)
    print("Write successful")

