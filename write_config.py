#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 17:37:04 2022

@author: lindsaygraff

Build config file
"""

import yaml

config_info = {
    'Geography': {
        'neighborhoods' : ['Hazelwood', 'Glen Hazel', 'Greenfield', 'Squirrel Hill South', 'Squirrel Hill North']
        },
    'Beta_Params': {
        'b_price': 1,
        'b_TT': 10/60,
        'b_risk_weight': 1,
        'b_disc_weight': 0.5,
        'b_rel_weight': 0.75
        },  # ultimately need b_rel_weight * b_TT
    'Speed_Params': {
        'walk_speed': 1.3,  # m/s
        'parking_speed': 5 / 3600 * 1609, # miles /hr / 3600 s/hr * 1609 meter/mile = m/s
        'scoot_speed': 15 / 3600 * 1000,    # 15 km/hr / (3600 s/hr ) * 1000 m/km = m/s
        'bike_speed': 15 / 3600 * 1000    # 15 km/hr / (3600 s/hr ) * 1000 m/km = m/s
        },   # km/hr
    'Price_Params': {
        'scoot_ppmin': 0.39,
        'bs_ppmin': 20/300,
        'TNC_fix_price': 1.51 + 1.60,
        'TNC_ppmile': 0.92,
        'TNC_ppmin': 0.34,
        'TNC_minfare_buffer': 8.32/4,
        'PT_price': 2.75,
        'pb_ppmin': 0,
        'pv_ppmile': 0.20,
        'zip_ppmin': 11/60,
        'zip_fixed_ppmonth': 9,
        'zip_est_num_trips': 4
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
    'Active_Mode_Parameters': {
        'active_modes': ['w','pb','bs','sc'],
        'nonactive_modes': ['pv','pt','t','z'],
        'discomf_weight_bike': 3/10,
        'discomf_weight_scoot_walk': 1/10,
        'rel_weight_active': 1,
        'rel_weight_nonactive': 1.5
        },
    'Risk_Parameters': {
        'risk_weight_active': 1.2,
        'crash_weight': 5
        }
    }

with open("config.yaml", 'w') as yamlfile:
    data = yaml.dump(config_info, yamlfile)
    print("Write successful")

